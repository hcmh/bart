#ifdef USE_CUDA
#ifdef USE_CUDNN

#include <complex.h>
#include <stdbool.h>
#include "cudnn.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/gpuops.h"
#include "num/blas_md_wrapper.h"
#include <cudnn.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "cudnn_wrapper.h"

static void cudnn_error(int line, cudnnStatus_t code)
{
	const char *err_str = cudnnGetErrorString(code);
	error("cudnn error: %d %s \n", line, err_str);
}


#define CUDNN_ERROR(x)	({ cudnnStatus_t errval = (x); if (CUDNN_STATUS_SUCCESS != errval) cudnn_error(__LINE__, errval); })


static cudnnHandle_t handle;
static bool handle_created = false;

static cudnnHandle_t get_handle(void)
{
	if (!handle_created)
		CUDNN_ERROR(cudnnCreate(&handle));
	handle_created = true;
	return handle;
}

#if 0
static void destroy_handle(void)
{
	CUDNN_ERROR(cudnnDestroy(handle));
	handle_created = false;
}
#endif


static bool check_trivial_cf_3d(int N, long odims[N], long ostrs[N], long idims[N], long istrs[N], long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], size_t size)
{
	if((28 != flags))
		return false;

	if ((NULL != dilation) && (!md_check_equal_dims(N, dilation, MD_SINGLETON_DIMS(N), ~(0l))))
		return false;
	if ((NULL != strides) && (!md_check_equal_dims(N, strides, MD_SINGLETON_DIMS(N), ~(0l))))
		return false;

	if (6 > N)
		return false;
	if (6 > md_calc_blockdim(N, odims, ostrs, size))
		return false;
	if (6 > md_calc_blockdim(N, idims, istrs, size))
		return false;
	if (6 > md_calc_blockdim(N, kdims, kstrs, size))
		return false;

	for (int i = 6; i< N; i++)
		if (1 != odims[i] * idims[i] * kdims[i])
			return false;

	//Check matmul dims
	if ((28 == flags) && ((1 != idims[0]) || (1 != odims[1]) || (1 != kdims[5])))
		return false;

	return true;
}

static bool check_trivial_cf_2d(int N, long odims[N], long ostrs[N], long idims[N], long istrs[N], long kdims[N], long kstrs[N],
				unsigned long flags, const long dilation[N], const long strides[N], size_t size)
{
	return check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size) && (1 == odims[4]) && (1 == idims[4]) && (1 == kdims[4]);
}


//Convert kernel bart channel first [out_channel, in_channel, imagex, ...] to NHWC [in_channel, image_x, ..., out_channel]
static void bart_real_kernel_to_cudnn_NHWC(int N, const long kdims[N], float* dst, const float* src)
{
	long trans_dims[2] = {md_calc_size(N - 1, kdims + 1), kdims[0]};
	long trans_strs_out[2] = {FL_SIZE, FL_SIZE * trans_dims[0]};
	long trans_strs_in[2] = {FL_SIZE * trans_dims[1], FL_SIZE};

	blas_smul_smatcopy(2, trans_dims, trans_strs_out, dst, trans_strs_in, src, 1.);
}

//Convert kernel NHWC [in_channel, image_x, ..., out_channel] to bart channel first [out_channel, in_channel, imagex, ...]
static void cudnn_NHWC_to_bart_real_kernel(int N, const long kdims[N], float* dst, const float* src)
{
	long trans_dims[2] = {kdims[0], md_calc_size(N - 1, kdims + 1)};
	long trans_strs_out[2] = {FL_SIZE, FL_SIZE * trans_dims[0]};
	long trans_strs_in[2] = {FL_SIZE * trans_dims[1], FL_SIZE};

	blas_smul_smatcopy(2, trans_dims, trans_strs_out, dst, trans_strs_in, src, 1.);
}

//Convert kernel bart channel first complex [out_channel, in_channel, imagex, ...] to NHWC real [2 * in_channel, image_x, ..., 2 * out_channel]
static void complex_kernel_to_real(int N, const long kdims[N], float* dst, const complex float* src)
{
	float* real = md_alloc_sameplace(N, kdims, FL_SIZE, dst);
	float* imag = md_alloc_sameplace(N, kdims, FL_SIZE, dst);

	md_real(N, kdims, real, src);
	md_imag(N, kdims, imag, src);

	long ckdims[N + 2];
	long rkdims[N + 2];

	for(int i = 0, ip = 0; ip < N + 2; ip++) {

		if ((ip == 0) || (ip == 2))
			continue;

		ckdims[ip] = kdims[i];
		rkdims[ip] = kdims[i];
		i++;
	}
	rkdims[0] = 2; rkdims[2] = 2;
	ckdims[0] = 1; ckdims[2] = 1;

	long rkstrs [N + 2]; md_calc_strides(N + 2, rkstrs, rkdims, FL_SIZE);
	long ckstrs [N + 2]; md_calc_strides(N + 2, ckstrs, ckdims, FL_SIZE);

	long pos[N + 2];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	float* tmp = md_alloc_sameplace(N + 2, rkdims, FL_SIZE, src);

	// re to re
	md_copy2(N + 2, ckdims, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), ckstrs, real, FL_SIZE);

	// im to im
	pos[0] = 1; pos[2] = 1;
	md_copy2(N + 2, ckdims, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), ckstrs, real, FL_SIZE);

	//re to im
	pos[0] = 1; pos[2] = 0;
	md_copy2(N + 2, ckdims, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), ckstrs, imag, FL_SIZE);

	//im to re
	pos[0] = 0; pos[2] = 1;
	md_smul(N + 2, ckdims, imag, imag, -1.);
	md_copy2(N + 2, ckdims, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), ckstrs, imag, FL_SIZE);

	md_free(real);
	md_free(imag);

	long tkdims[N]; md_copy_dims(N, tkdims, kdims);
	tkdims[0] *= 2;
	tkdims[1] *= 2;

	bart_real_kernel_to_cudnn_NHWC(N, tkdims, dst, tmp);
}

//Convert kernel NHWC real [2 * in_channel, image_x, ..., 2 * out_channel] to bart channel first complex [out_channel, in_channel, imagex, ...] (adjoint ooperation)
static void complex_kernel_to_real_adjoint_cf(int N, const long kdims[N], complex float* dst, const float* src)
{
	long tkdims[N]; md_copy_dims(N, tkdims, kdims);
	tkdims[0] *= 2;
	tkdims[1] *= 2;
	float* tmp = md_alloc_gpu(N, tkdims, FL_SIZE);

	cudnn_NHWC_to_bart_real_kernel(N, tkdims, tmp, src);

	long ckdims[N + 2];
	long rkdims[N + 2];

	for(int i = 0, ip = 0; ip < N + 2; ip++) {

		if ((ip == 0) || (ip == 2))
			continue;

		ckdims[ip] = kdims[i];
		rkdims[ip] = kdims[i];
		i++;
	}
	rkdims[0] = 2; rkdims[2] = 2;
	ckdims[0] = 1; ckdims[2] = 1;

	long rkstrs [N + 2]; md_calc_strides(N + 2, rkstrs, rkdims, FL_SIZE);
	long ckstrs [N + 2]; md_calc_strides(N + 2, ckstrs, ckdims, FL_SIZE);

	long pos[N + 2];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	float* real = md_alloc_sameplace(N, kdims, FL_SIZE, dst);
	float* imag = md_alloc_sameplace(N, kdims, FL_SIZE, dst);
	float* tmp1 = md_alloc_sameplace(N, kdims, FL_SIZE, dst);

	md_copy2(N + 2, ckdims, ckstrs, real, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), FL_SIZE);

	pos[0] = 1; pos[2] = 1;
	md_copy2(N + 2, ckdims, ckstrs, tmp1, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), FL_SIZE);
	md_add(N + 2, ckdims, real, real, tmp1);

	pos[0] = 1; pos[2] = 0;
	md_copy2(N + 2, ckdims, ckstrs, imag, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), FL_SIZE);

	pos[0] = 0; pos[2] = 1;
	md_copy2(N + 2, ckdims, ckstrs, tmp1, rkstrs, (void*)tmp + md_calc_offset(N + 2, rkstrs, pos), FL_SIZE);
	md_sub(N + 2, ckdims, imag, imag, tmp1);
	md_free(tmp1);
	md_free(tmp);

	md_zcmpl(N, kdims, dst, real, imag);

	md_free(real);
	md_free(imag);
}

static void cudnn_frw_in_2d_real(	long OC, long IC, long OX, long OY, long IX, long IY, long KX, long KY, long NB,
				float* out, const float* in, const float* krn, bool conv,
				cudnnTensorFormat_t tensor_format)
{
	cudnnTensorDescriptor_t in_desc;
	cudnnTensorDescriptor_t out_desc;
	cudnnFilterDescriptor_t krn_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc));
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&krn_desc));

	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc, tensor_format, CUDNN_DATA_FLOAT, NB, IC, IY, IX));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc, tensor_format, CUDNN_DATA_FLOAT, NB, OC, OY, OX));
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(krn_desc, CUDNN_DATA_FLOAT, tensor_format, OC, IC, KY, KX));

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t algo;
	CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(get_handle(), in_desc, krn_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

	size_t ws_size = 0;
	CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(get_handle(), in_desc, krn_desc, conv_desc, out_desc, algo, &ws_size));
	void* workspace = (0 < ws_size) ? cuda_malloc(ws_size) : NULL;

	float alpha = 1.;
	float beta = 1.;
	CUDNN_ERROR(cudnnConvolutionForward(get_handle(), &alpha, in_desc, in, krn_desc, krn, conv_desc, algo, workspace, ws_size, &beta, out_desc, out));

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(krn_desc));
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
}

static void cudnn_bwd_krn_2d_real(	long OC, long IC, long OX, long OY, long IX, long IY, long KX, long KY, long NB,
				const float* out, const float* in, float* krn, bool conv,
				cudnnTensorFormat_t tensor_format)
{
	cudnnTensorDescriptor_t in_desc;
	cudnnTensorDescriptor_t out_desc;
	cudnnFilterDescriptor_t krn_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc));
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&krn_desc));

	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc, tensor_format, CUDNN_DATA_FLOAT, NB, IC, IY, IX));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc, tensor_format, CUDNN_DATA_FLOAT, NB, OC, OY, OX));
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(krn_desc, CUDNN_DATA_FLOAT, tensor_format, OC, IC, KY, KX));

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionBwdFilterAlgo_t algo;
	CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(get_handle(), in_desc, out_desc, conv_desc, krn_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));

	size_t ws_size = 0;
	CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(get_handle(), in_desc, out_desc, conv_desc, krn_desc, algo, &ws_size));
	void* workspace = (0 < ws_size) ? cuda_malloc(ws_size) : NULL;

	float alpha = 1.;
	float beta = 1.;
	CUDNN_ERROR(cudnnConvolutionBackwardFilter(get_handle(), &alpha, in_desc, in, out_desc, out, conv_desc, algo, workspace, ws_size, &beta, krn_desc, krn));
	md_free(workspace);

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(krn_desc));
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
}

static void cudnn_bwd_in_2d_real(	long OC, long IC, long OX, long OY, long IX, long IY, long KX, long KY, long NB,
				const float* out, float* in, const float* krn, bool conv,
				cudnnTensorFormat_t tensor_format)
{
	cudnnTensorDescriptor_t in_desc;
	cudnnTensorDescriptor_t out_desc;
	cudnnFilterDescriptor_t krn_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc));
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&krn_desc));

	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc, tensor_format, CUDNN_DATA_FLOAT, NB, IC, IY, IX));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc, tensor_format, CUDNN_DATA_FLOAT, NB, OC, OY, OX));
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(krn_desc, CUDNN_DATA_FLOAT, tensor_format, OC, IC, KY, KX));

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(get_handle(), krn_desc, out_desc, conv_desc, in_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));

	size_t ws_size = 0;
	CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(get_handle(), krn_desc, out_desc, conv_desc, in_desc, algo, &ws_size));
	void* workspace = (0 < ws_size) ? cuda_malloc(ws_size) : NULL;

	float alpha = 1.;
	float beta = 1.;
	CUDNN_ERROR(cudnnConvolutionBackwardData(get_handle(), &alpha, krn_desc, krn, out_desc, out, conv_desc, algo, workspace, ws_size, &beta, in_desc, in));
	md_free(workspace);

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(krn_desc));
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
}

static void cudnn_tensor_transform(cudnnTensorFormat_t format_out, cudnnTensorFormat_t format_in, long N, long C, long H, long W, float* dst, const float* src)
{
	cudnnTensorDescriptor_t in_desc;
	cudnnTensorDescriptor_t out_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc));

	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc, format_in, CUDNN_DATA_FLOAT, N, C, H, W));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc, format_out, CUDNN_DATA_FLOAT, N, C, H, W));

	float alpha = 1.;
	float beta = 1.;

	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, in_desc, src, &beta, out_desc, dst));

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc));
}


bool zconvcorr_fwd_cudnn_2d_cf(	int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (!check_trivial_cf_2d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
		return false;

	float* krn_tmp = cuda_malloc(4 * md_calc_size(N, kdims) * FL_SIZE);
	complex_kernel_to_real(N, kdims, krn_tmp, krn);

	cudnn_frw_in_2d_real(	2 * odims[0], 2 * idims[1], odims[2], odims[3], idims[2], idims[3], kdims[2], kdims[3], odims[5],
				(float*)out, (const float*)in, krn_tmp, conv, CUDNN_TENSOR_NHWC);
	md_free(krn_tmp);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

bool zconvcorr_bwd_in_cudnn_2d_cf(	int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (!check_trivial_cf_2d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
		return false;

	complex float* krn_conj = md_alloc_gpu(N, kdims, CFL_SIZE);
	md_zconj(N, kdims, krn_conj, krn);

	float* krn_tmp = cuda_malloc(4 * md_calc_size(N, kdims) * FL_SIZE);
	complex_kernel_to_real(N, kdims, krn_tmp, krn_conj);

	md_free(krn_conj);

	if (false) { //for some cases this is slow

		cudnn_bwd_in_2d_real(	2 * odims[0], 2 * idims[1], odims[2], odims[3], idims[2], idims[3], kdims[2], kdims[3], odims[5],
					(const float*)out, (float*)in, krn_tmp, conv, CUDNN_TENSOR_NHWC);
	} else {

		float* in_tmp2 = cuda_malloc(md_calc_size(N, idims) * CFL_SIZE);
		float* out_tmp2 = cuda_malloc(md_calc_size(N, odims) * CFL_SIZE);
		float* krn_tmp2 = cuda_malloc(4 * md_calc_size(N, kdims) * FL_SIZE);
		cuda_clear(md_calc_size(N, idims) * CFL_SIZE, in_tmp2);
		cuda_clear(md_calc_size(N, odims) * CFL_SIZE, out_tmp2);
		cuda_clear(4 * md_calc_size(N, kdims) * FL_SIZE, krn_tmp2);

		cudnn_tensor_transform(CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC, odims[5], 2 * odims[0], odims[3], odims[2], out_tmp2, (const float*)out);
		cudnn_tensor_transform(CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC, 2 * kdims[0], 2 * kdims[1], kdims[3], kdims[2], krn_tmp2, (const float*)krn_tmp);

		cudnn_bwd_in_2d_real(	2 * odims[0], 2 * idims[1], odims[2], odims[3], idims[2], idims[3], kdims[2], kdims[3], odims[5],
					(const float*)out_tmp2, (float*)in_tmp2, krn_tmp2, conv, CUDNN_TENSOR_NCHW);

		cudnn_tensor_transform(CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW, idims[5], 2 * idims[1], idims[3], idims[2], (float*)in, (const float*)in_tmp2);

		md_free(krn_tmp2);
		md_free(in_tmp2);
		md_free(out_tmp2);
	}

	md_free(krn_tmp);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

bool zconvcorr_bwd_krn_cudnn_2d_cf(	int N,
				long odims[N], long ostrs[N], const complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (!check_trivial_cf_2d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
		return false;

	complex float* in_conj = md_alloc_gpu(N, idims, CFL_SIZE);
	md_zconj(N, idims, in_conj, in);


	float* krn_tmp = cuda_malloc(4 * md_calc_size(N, kdims) * FL_SIZE);
	cuda_clear(4 * md_calc_size(N, kdims) * FL_SIZE, krn_tmp);

	cudnn_bwd_krn_2d_real(	2 * odims[0], 2 * idims[1], odims[2], odims[3], idims[2], idims[3], kdims[2], kdims[3], odims[5],
				(const float*)out, (const float*)in_conj, krn_tmp, conv, CUDNN_TENSOR_NHWC);

	md_free(in_conj);

	complex float* krn_tmp2 = md_alloc_gpu(N, kdims, CFL_SIZE);
	complex_kernel_to_real_adjoint_cf(N, kdims, krn_tmp2, krn_tmp);
	md_free(krn_tmp);

	md_zadd(N, kdims, krn, krn, krn_tmp2);
	md_free(krn_tmp2);
	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

#endif
#endif
