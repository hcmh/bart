#ifdef USE_CUDNN
#include <complex.h>
#include <stdbool.h>
#include "cudnn.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/gpuops.h"
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

static cudnnHandle_t destroy_handle(void)
{
	CUDNN_ERROR(cudnnDestroy(handle));
	handle_created = false;
}


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

bool zconvcorr_fwd_cudnn_2d_cf(	int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (!check_trivial_cf_2d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
		return false;

	cudnnTensorDescriptor_t in_desc;
	cudnnTensorDescriptor_t out_desc;
	cudnnTensorDescriptor_t krn_desc_cf;
	cudnnTensorDescriptor_t krn_desc_cl;
	cudnnFilterDescriptor_t krn_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cf));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cl));
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&krn_desc));

	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, idims[5], 2 * idims[1], idims[3], idims[2]));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, odims[5], 2 * odims[0], odims[3], odims[2]));
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(krn_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2 * kdims[0], 2 * kdims[1], kdims[3], kdims[2]));

	int kdims_cudnn_trans[4] = {kdims[0], kdims[1], kdims[3], kdims[2]};
	int kstrs_cudnn_in[4] = {2 * kstrs[0] / size, 2 * kstrs[1] / size, 2 * kstrs[3] / size, 2 * kstrs[2] / size};
	int kstrs_cudnn_out[4] = {4 * kdims_cudnn_trans[1] * kdims_cudnn_trans[2] * kdims_cudnn_trans[3],  2 * kdims_cudnn_trans[2] * kdims_cudnn_trans[3], kdims_cudnn_trans[3], 1};

	for (int i = 0; i < 4; i++) {

		kstrs_cudnn_in[i] = MAX(1, kstrs_cudnn_in[i]);
		kstrs_cudnn_out[i] = MAX(1, kstrs_cudnn_out[i]);
	}


	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cf, CUDNN_DATA_FLOAT, 4, kdims_cudnn_trans, kstrs_cudnn_in));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cl, CUDNN_DATA_FLOAT, 4, kdims_cudnn_trans, kstrs_cudnn_out));

	float* krn_tmp = md_alloc_gpu(N, kdims, 2 * size);
	float alpha = 1;
	float beta = 0;

	//real of filter
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, krn, &beta, krn_desc_cl, krn_tmp)); // re -> re
	alpha = 1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, krn, &beta, krn_desc_cl, krn_tmp + kdims_cudnn_trans[3] * kdims_cudnn_trans[2] + 2 * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1])); // im -> im

	//imag of filter
	alpha = 1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, (float*)krn + 1, &beta, krn_desc_cl, krn_tmp + 2 * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1])); // re -> im
	alpha = -1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, (float*)krn + 1, &beta, krn_desc_cl, krn_tmp + kdims_cudnn_trans[3] * kdims_cudnn_trans[2])); // im -> re
	alpha = 1.;

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t algo;
	CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(get_handle(), in_desc, krn_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

	size_t ws_size = 0;
	CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(get_handle(), in_desc, krn_desc, conv_desc, out_desc, algo, &ws_size));
	void* workspace = (0 < ws_size) ? cuda_malloc(ws_size) : NULL;

	alpha = 1.;
	beta = 1.;
	CUDNN_ERROR(cudnnConvolutionForward(get_handle(), &alpha, in_desc, in, krn_desc, krn_tmp, conv_desc, algo, workspace, ws_size, &beta, out_desc, out));

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cf));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cl));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(krn_desc));
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
	md_free(workspace);
	md_free(krn_tmp);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

bool zconvcorr_fwd_cudnn_3d_cf(	int N,
				long odims[N], long ostrs[N], complex float* out,
				long idims[N], long istrs[N], const complex float* in,
				long kdims[N], long kstrs[N], const complex float* krn,
				unsigned long flags, const long dilation[N], const long strides[N], bool conv)
{
	size_t size = CFL_SIZE;

	if (!check_trivial_cf_3d(N, odims, ostrs, idims, istrs, kdims, kstrs, flags, dilation, strides, size))
		return false;

	cudnnTensorDescriptor_t in_desc;
	cudnnTensorDescriptor_t out_desc;
	cudnnTensorDescriptor_t krn_desc_cf;
	cudnnTensorDescriptor_t krn_desc_cl;
	cudnnFilterDescriptor_t krn_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cf));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cl));
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&krn_desc));

	int cudnn_idims[5] = {idims[5], 2 * idims[1], idims[4], idims[3], idims[2]};
	int cudnn_istrs[5] = {	2 * idims[1] * idims[2] * idims[3] * idims[4],
				1,
				2 * idims[1] * idims[2] * idims[3],
				2 * idims[1] * idims[2],
				2 * idims[1],
				};
	int cudnn_ostrs[5] = {	2 * odims[0] * odims[2] * odims[3] * odims[4],
				1,
				2 * odims[0] * odims[2] * odims[3],
				2 * odims[0] * odims[2],
				2 * odims[0],
				};
	int cudnn_odims[5] = {odims[5], 2 * odims[0], odims[4], odims[3], odims[2]};
	int cudnn_kdims[5] = {2 * kdims[0], 2 * kdims[1], kdims[4], kdims[3], kdims[2]};

	CUDNN_ERROR(cudnnSetTensorNdDescriptor(in_desc, CUDNN_DATA_FLOAT, 5, cudnn_idims, cudnn_istrs));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(out_desc, CUDNN_DATA_FLOAT, 5, cudnn_odims, cudnn_ostrs));
	CUDNN_ERROR(cudnnSetFilterNdDescriptor(krn_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, cudnn_kdims));

	int kdims_cudnn_trans[5] = {kdims[0], kdims[1], kdims[4], kdims[3], kdims[2]};
	int kstrs_cudnn_in[5] = {2 * kstrs[0] / size, 2 * kstrs[1] / size, 2 * kstrs[4] / size, 2 * kstrs[3] / size, 2 * kstrs[2] / size};
	int kstrs_cudnn_out[5] = {	4 * kdims_cudnn_trans[1] * kdims_cudnn_trans[2] * kdims_cudnn_trans[3] * kdims_cudnn_trans[4],
					2 * kdims_cudnn_trans[2] * kdims_cudnn_trans[3] * kdims_cudnn_trans[4],
					kdims_cudnn_trans[3] * kdims_cudnn_trans[4],
					kdims_cudnn_trans[4],
					1};

	for (int i = 0; i < 5; i++) {

		kstrs_cudnn_in[i] = MAX(1, kstrs_cudnn_in[i]);
		kstrs_cudnn_out[i] = MAX(1, kstrs_cudnn_out[i]);
	}


	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cf, CUDNN_DATA_FLOAT, 5, kdims_cudnn_trans, kstrs_cudnn_in));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cl, CUDNN_DATA_FLOAT, 5, kdims_cudnn_trans, kstrs_cudnn_out));

	float* krn_tmp = md_alloc_gpu(N, kdims, 2 * size);
	float alpha = 1;
	float beta = 0;

	//real of filter
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, krn, &beta, krn_desc_cl, krn_tmp)); // re -> re
	alpha = 1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, krn, &beta, krn_desc_cl, krn_tmp +	kdims_cudnn_trans[4] * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] + 2 * kdims_cudnn_trans[4] * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1])); // im -> im

	//imag of filter
	alpha = 1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, (float*)krn + 1, &beta, krn_desc_cl, krn_tmp + 2 * kdims_cudnn_trans[4] *  kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1])); // re -> im
	alpha = -1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, (float*)krn + 1, &beta, krn_desc_cl, krn_tmp + kdims_cudnn_trans[4] * kdims_cudnn_trans[3] * kdims_cudnn_trans[2])); // im -> re
	alpha = 1.;

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_ERROR(cudnnSetConvolutionNdDescriptor(conv_desc, 3, MAKE_ARRAY(0,0,0), MAKE_ARRAY(1,1,1), MAKE_ARRAY(1,1,1), conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t algo;
	CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(get_handle(), in_desc, krn_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

	size_t ws_size = 0;
	CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(get_handle(), in_desc, krn_desc, conv_desc, out_desc, algo, &ws_size));
	void* workspace = (0 < ws_size) ? cuda_malloc(ws_size) : NULL;

	alpha = 1.;
	beta = 1.;
	CUDNN_ERROR(cudnnConvolutionForward(get_handle(), &alpha, in_desc, in, krn_desc, krn_tmp, conv_desc, algo, workspace, ws_size, &beta, out_desc, out));

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cf));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cl));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(krn_desc));
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
	md_free(workspace);
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

	cudnnTensorDescriptor_t in_desc;
	cudnnTensorDescriptor_t out_desc_NHWC;
	cudnnTensorDescriptor_t out_desc_NCHW;
	cudnnTensorDescriptor_t krn_desc_cf;
	cudnnTensorDescriptor_t krn_desc_cl;
	cudnnFilterDescriptor_t krn_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc_NCHW));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc_NHWC));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cf));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cl));
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&krn_desc));

	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, idims[5], 2 * idims[1], idims[3], idims[2]));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc_NCHW, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, odims[5], 2 * odims[0], odims[3], odims[2]));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc_NHWC, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, odims[5], 2 * odims[0], odims[3], odims[2]));
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(krn_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2 * kdims[0], 2 * kdims[1], kdims[3], kdims[2]));

	int kdims_cudnn_trans[4] = {kdims[0], kdims[1], kdims[3], kdims[2]};
	int kstrs_cudnn_in[4] = {2 * kstrs[0] / size, 2 * kstrs[1] / size, 2 * kstrs[3] / size, 2 * kstrs[2] / size};
	int kstrs_cudnn_out[4] = {4 * kdims_cudnn_trans[1] * kdims_cudnn_trans[2] * kdims_cudnn_trans[3],  2 * kdims_cudnn_trans[2] * kdims_cudnn_trans[3],  kdims_cudnn_trans[3], 1};

	for (int i = 0; i < 4; i++) {

		kstrs_cudnn_in[i] = MAX(1, kstrs_cudnn_in[i]);
		kstrs_cudnn_out[i] = MAX(1, kstrs_cudnn_out[i]);
	}


	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cf, CUDNN_DATA_FLOAT, 4, kdims_cudnn_trans, kstrs_cudnn_in));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cl, CUDNN_DATA_FLOAT, 4, kdims_cudnn_trans, kstrs_cudnn_out));

	float* krn_tmp = md_alloc_gpu(N, kdims, 2 * size);
	float alpha = 1;
	float beta = 0;

	//real of filter
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, krn, &beta, krn_desc_cl, krn_tmp)); // re -> re
	alpha = 1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, krn, &beta, krn_desc_cl, krn_tmp + kdims_cudnn_trans[2] * kdims_cudnn_trans[3] + 2 * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1])); // im -> im

	//imag of filter
	alpha = -1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, (float*)krn + 1, &beta, krn_desc_cl, krn_tmp + 2 * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1])); // re -> im
	alpha = 1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cf, (float*)krn + 1, &beta, krn_desc_cl, krn_tmp + kdims_cudnn_trans[2] * kdims_cudnn_trans[3])); // im -> re
	alpha = 1.;

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionBwdDataAlgo_t algo;
	CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(get_handle(), krn_desc, out_desc_NCHW, conv_desc, in_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));

	size_t ws_size = 0;
	CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(get_handle(), krn_desc, out_desc_NCHW, conv_desc, in_desc, algo, &ws_size));
	void* workspace = (0 < ws_size) ? cuda_malloc(ws_size) : NULL;

	float* out_tmp = md_alloc_gpu(N, odims, CFL_SIZE);
	alpha = 1.;
	beta = 0.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, out_desc_NHWC, out, &beta, out_desc_NCHW, out_tmp));

	alpha = 1.;
	beta = 1.;
	CUDNN_ERROR(cudnnConvolutionBackwardData(get_handle(), &alpha, krn_desc, krn_tmp, out_desc_NCHW, out_tmp, conv_desc, algo, workspace, ws_size, &beta, in_desc, in));

	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc_NCHW));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc_NHWC));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cf));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cl));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(krn_desc));
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
	md_free(workspace);
	md_free(out_tmp);
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

	cudnnTensorDescriptor_t in_desc_NCHW;
	cudnnTensorDescriptor_t in_desc_NHWC;
	cudnnTensorDescriptor_t out_desc_NHWC;
	cudnnTensorDescriptor_t out_desc_NCHW;
	cudnnTensorDescriptor_t krn_desc_cf;
	cudnnTensorDescriptor_t krn_desc_cl;
	cudnnFilterDescriptor_t krn_desc;

	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc_NCHW));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&in_desc_NHWC));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc_NCHW));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&out_desc_NHWC));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cf));
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&krn_desc_cl));
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&krn_desc));

	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc_NHWC, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, idims[5], 2 * idims[1], idims[3], idims[2]));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(in_desc_NCHW, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, idims[5], 2 * idims[1], idims[3], idims[2]));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc_NCHW, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, odims[5], 2 * odims[0], odims[3], odims[2]));
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(out_desc_NHWC, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, odims[5], 2 * odims[0], odims[3], odims[2]));
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(krn_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2 * kdims[0], 2 * kdims[1], kdims[3], kdims[2]));

	float* krn_tmp = md_alloc_gpu(N, kdims, 2 * size);
	float* out_tmp = md_alloc_gpu(N, odims, size);
	float* in_tmp = md_alloc_gpu(N, idims, size);

	float alpha = 1.;
	float beta = 0.;

	CUDNN_ERROR(cudnnTransformTensor(get_handle(),&alpha, in_desc_NHWC, in, &beta, in_desc_NCHW, in_tmp));
	CUDNN_ERROR(cudnnTransformTensor(get_handle(),&alpha, out_desc_NHWC, out, &beta, out_desc_NCHW, out_tmp));

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, conv ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	cudnnConvolutionBwdFilterAlgo_t algo;
	CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(get_handle(), in_desc_NCHW, out_desc_NCHW, conv_desc, krn_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));

	size_t ws_size = 0;
	CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(get_handle(), in_desc_NCHW, out_desc_NCHW, conv_desc, krn_desc, algo, &ws_size));
	void* workspace = (0 < ws_size) ? cuda_malloc(ws_size) : NULL;

	alpha = 1.;
	beta = 0.;
	CUDNN_ERROR(cudnnConvolutionBackwardFilter(get_handle(), &alpha, in_desc_NCHW, in_tmp, out_desc_NCHW, out_tmp, conv_desc, algo, workspace, ws_size, &beta, krn_desc, krn_tmp));


	int kdims_cudnn_trans[4] = {kdims[0], kdims[1], kdims[3], kdims[2]};
	int kstrs_cudnn_in[4] = {2 * kstrs[0] / size, 2 * kstrs[1] / size, 2 * kstrs[3] / size, 2 * kstrs[2] / size};
	int kstrs_cudnn_out[4] = {4 * kdims_cudnn_trans[1] * kdims_cudnn_trans[2] * kdims_cudnn_trans[3],  2 * kdims_cudnn_trans[2] * kdims_cudnn_trans[3],  kdims_cudnn_trans[3], 1};

	for (int i = 0; i < 4; i++) {

		kstrs_cudnn_in[i] = MAX(1, kstrs_cudnn_in[i]);
		kstrs_cudnn_out[i] = MAX(1, kstrs_cudnn_out[i]);
	}


	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cf, CUDNN_DATA_FLOAT, 4, kdims_cudnn_trans, kstrs_cudnn_in));
	CUDNN_ERROR(cudnnSetTensorNdDescriptor(krn_desc_cl, CUDNN_DATA_FLOAT, 4, kdims_cudnn_trans, kstrs_cudnn_out));

	alpha = 1;
	beta = 1;

	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cl, krn_tmp, &beta, krn_desc_cf, krn)); // re -> re
	alpha = -1;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cl, krn_tmp + kdims_cudnn_trans[2] * kdims_cudnn_trans[3] + 2 * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1], &beta, krn_desc_cf, krn)); // im -> im
	alpha = 1.;
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cl, krn_tmp + 2 * kdims_cudnn_trans[3] * kdims_cudnn_trans[2] * kdims_cudnn_trans[1] , &beta, krn_desc_cf, (float*)krn + 1)); // re -> im
	CUDNN_ERROR(cudnnTransformTensor(get_handle(), &alpha, krn_desc_cl, krn_tmp + kdims_cudnn_trans[2] * kdims_cudnn_trans[3], &beta, krn_desc_cf, (float*)krn + 1)); // im -> re


	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc_NHWC));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(in_desc_NCHW));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc_NCHW));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(out_desc_NHWC));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cf));
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(krn_desc_cl));
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(krn_desc));
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
	md_free(workspace);
	md_free(in_tmp);
	md_free(out_tmp);
	md_free(krn_tmp);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}

#endif //USE_CUDNN