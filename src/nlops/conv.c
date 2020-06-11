/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <assert.h>
#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "linops/someops.h"
#include "linops/linop.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/nlop.h"

#include "conv.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif


static struct nlop_s* nlop_resize_create(long N, const long out_dims[N], const long in_dims[N], bool center)//
{
	auto lresize = (center ? linop_resize_center_create : linop_resize_create)(N, out_dims, in_dims);
	auto nresize = nlop_from_linop(lresize);
	linop_free(lresize);
	return nresize;
}


struct convcorr_geom_s {

	INTERFACE(nlop_data_t);

	long N;

	const long* odims;
	const long* idims1;
	const long* idims2;

	const long* mdims;
	const long* ostrs;
	const long* istrs1;
	const long* istrs2;

	unsigned int flags;

	long shift;
	complex float* src1;
	complex float* src2;
};

DEF_TYPEID(convcorr_geom_s);

static void convcorr_initialize(struct convcorr_geom_s* data, const complex float* arg, bool der1, bool der2)
{
	if (der2 && (NULL == data->src1))
		data->src1 = md_alloc_sameplace(data->N, data->idims1, CFL_SIZE, arg);
	
	if (!der2 && (NULL != data->src1)) {

		md_free(data->src1);
		data->src1 = NULL;
	}
		
	if (der1 && (NULL == data->src2))
		data->src2 = md_alloc_sameplace(data->N, data->idims2, CFL_SIZE, arg);

	if (!der1 && (NULL != data->src2)) {

		md_free(data->src2);
		data->src2 = NULL;
	}
}

static void convcorr_geom_fun(const nlop_data_t* _data, int N, complex float* args[N], operator_run_opt_flags_t run_flags[N][N])
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif

	bool der1 = !(MD_IS_SET(run_flags[0][1], OP_APP_NO_DER));
	bool der2 = !(MD_IS_SET(run_flags[0][2], OP_APP_NO_DER));

	convcorr_initialize(data, dst, der1, der2);

	//conj to have benefits of fmac optimization in adjoints
	if(der2)
		md_zconj(data->N, data->idims1, data->src1, src1);
	if(der1)
		md_zconj(data->N, data->idims2, data->src2, src2);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs1, src1, data->istrs2, src2 + data->shift);

	PRINT_TIMER("frw convgeo");
}


static void convcorr_geom_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if (NULL == data->src1)
		error("Convcorr %x derivative not available\n", data);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmacc2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs2, src + data->shift, data->istrs1, data->src1);

	PRINT_TIMER("der2 convgeo");
}

static void convcorr_geom_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if (NULL == data->src1)
		error("Convcorr %x derivative not available\n", data);

	md_clear(data->N, data->idims2, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, data->mdims, data->istrs2, dst + data->shift, data->ostrs, src, data->istrs1, data->src1);

	PRINT_TIMER("adj2 convgeo");
}

static void convcorr_geom_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if (NULL == data->src2)
		error("Convcorr %x derivative not available\n", data);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmacc2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs1, src, data->istrs2, data->src2 + data->shift);

	PRINT_TIMER("der1 convgeo");
}

static void convcorr_geom_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if (NULL == data->src2)
		error("Convcorr %x derivative not available\n", data);

	md_clear(data->N, data->idims1, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, data->mdims, data->istrs1, dst, data->ostrs, src, data->istrs2, data->src2 + data->shift);

	PRINT_TIMER("adj1 convgeo");
}


static void convcorr_geom_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	md_free(data->src1);
	md_free(data->src2);

	xfree(data->odims);
	xfree(data->idims1);
	xfree(data->idims2);

	xfree(data->mdims);
	xfree(data->ostrs);
	xfree(data->istrs1);
	xfree(data->istrs2);

	xfree(data);
}

static struct nlop_s* nlop_convcorr_geom_valid_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], bool conv, bool transp)
{
	for (int i = 0; i < N; i++)
		if MD_IS_SET(flags, i)
			assert(odims[i] == idims[i] - (kdims[i] - 1));

	PTR_ALLOC(struct convcorr_geom_s, data);
	SET_TYPEID(convcorr_geom_s, data);

	data->flags = flags;

	// will be initialized later, to transparently support GPU
	data->src1 = NULL;
	data->src2 = NULL;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], transp ? idims : odims);

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], transp ? odims : idims);
	md_copy_dims(N, nl_idims[1], kdims);

	data->N = N;

	PTR_ALLOC(long[N], nodims);
	PTR_ALLOC(long[N], nidims1);
	PTR_ALLOC(long[N], nidims2);

	PTR_ALLOC(long[2 * N], nmdims);
	PTR_ALLOC(long[2 * N], nostrs);
	PTR_ALLOC(long[2 * N], nistrs1);
	PTR_ALLOC(long[2 * N], nistrs2);

	md_copy_dims(N, *nodims, nl_odims[0]);
	md_copy_dims(N, *nidims1, nl_idims[0]);
	md_copy_dims(N, *nidims2, nl_idims[1]);

	if (transp)
		data->shift = calc_convcorr_geom(	N, flags,
							*nmdims, *nistrs1, *nistrs2, *nostrs,
							odims, MD_STRIDES(N, odims, CFL_SIZE),
							kdims, MD_STRIDES(N, kdims, CFL_SIZE),
							idims, MD_STRIDES(N, idims, CFL_SIZE),
							conv) / CFL_SIZE;
	else
		data->shift = calc_convcorr_geom(	N, flags,
							*nmdims, *nostrs, *nistrs2, *nistrs1,
							odims, MD_STRIDES(N, odims, CFL_SIZE),
							kdims, MD_STRIDES(N, kdims, CFL_SIZE),
							idims, MD_STRIDES(N, idims, CFL_SIZE),
							conv) / CFL_SIZE;

	data->odims = *PTR_PASS(nodims);
	data->idims1 = *PTR_PASS(nidims1);
	data->idims2 = *PTR_PASS(nidims2);

	data->mdims = *PTR_PASS(nmdims);
	data->ostrs = *PTR_PASS(nostrs);
	data->istrs1 = *PTR_PASS(nistrs1);
	data->istrs2 = *PTR_PASS(nistrs2);

	operator_prop_flags_t props[2][1] = {{MD_BIT(OP_PROP_C_LIN)}, {MD_BIT(OP_PROP_C_LIN)}};

	return nlop_generic_extopts_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), convcorr_geom_fun,
				    (nlop_fun_t[2][1]){ { convcorr_geom_der1 }, { convcorr_geom_der2 } },
				    (nlop_fun_t[2][1]){ { convcorr_geom_adj1 }, { convcorr_geom_adj2 } }, NULL, NULL, convcorr_geom_del, props);
}


struct nlop_s* nlop_convcorr_geom_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum PADDING conv_pad, bool conv, char transpc)
{
	struct nlop_s* result = NULL;

	assert(('N' == transpc) || ('T' == transpc) || ('C' == transpc));

	bool transp = ('N' != transpc);
	
	if (PAD_VALID == conv_pad) {

		result = nlop_convcorr_geom_valid_create(N, flags, odims, idims, kdims, conv, transp);
	} else {

		long pad[N];
		long nidims[N];
		for (int i = 0; i < N; i++) {

			if(MD_IS_SET(flags, i)) {

				assert(1 == kdims[i] % 2);
				pad[i] = kdims[i] / 2;
			} else {

				pad[i] = 0;		
			}
			nidims[i] = idims[i] + 2 * pad[i];
		}

		result = nlop_convcorr_geom_valid_create(N, flags, odims, nidims, kdims, conv, transp);
		
		struct linop_s* pad_op = linop_padding_create(N, idims, conv_pad, pad, pad);

		if (transp) {

			result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_adjoint(pad_op)), 0);
			linop_free(pad_op);
		} else {

			result = nlop_chain2_FF(nlop_from_linop_F(pad_op), 0, result, 0);
			result = nlop_permute_inputs_F(result, 2, MAKE_ARRAY(1, 0));
		}
	}

	if ('C' == transpc)
		result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, kdims)), 0, result, 1);

	return result;
}


struct conv_fft_s {

	INTERFACE(nlop_data_t);
	long N;
	unsigned int flags;

	const long* odims;
	const long* idims;
	const long* kdims;

	const long* kdims2;

	complex float* image;
	complex float* kernel;

	complex float scaling;
};

DEF_TYPEID(conv_fft_s);


static void conv_fft_initialize(struct conv_fft_s* data, const complex float* arg)
{
	if (NULL == data->image)
		data->image = md_alloc_sameplace(data->N, data->idims, CFL_SIZE, arg);

	if (NULL == data->kernel)
		data->kernel = md_alloc_sameplace(data->N, data->kdims, CFL_SIZE, arg);
}


static void conv_fft_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(conv_fft_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif
	conv_fft_initialize(data, dst);

	ifft(data->N, data->idims, data->flags, data->image, src1);
	md_zsmul(data->N, data->kdims, data->kernel, src2, data->scaling);

	complex float* kernel_tmp = NULL;

	kernel_tmp = md_alloc_sameplace(data->N, data->kdims2, CFL_SIZE, data->kernel);
	md_resize_center(data->N, data->kdims2, kernel_tmp, data->kdims, data->kernel, CFL_SIZE);

	ifft(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);
	ifftmod(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);

	md_ztenmul(data->N, data->odims, dst, data->idims, data->image, data->kdims2, kernel_tmp);
	md_free(kernel_tmp);

	fft(data->N, data->odims, data->flags, dst, dst);
}

static void conv_fft_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(conv_fft_s, _data);

	complex float* kernel_tmp2 = NULL;

	kernel_tmp2 = md_alloc_sameplace(data->N, data->kdims, CFL_SIZE, data->kernel);
	md_zsmul(data->N, data->kdims, kernel_tmp2, src, data->scaling);

	complex float* kernel_tmp = NULL;

	kernel_tmp = md_alloc_sameplace(data->N, data->kdims2, CFL_SIZE, data->kernel);
	md_resize_center(data->N, data->kdims2, kernel_tmp, data->kdims, kernel_tmp2, CFL_SIZE);
	md_free(kernel_tmp2);

	ifft(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);
	ifftmod(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);

	md_ztenmul(data->N, data->odims, dst, data->idims, data->image, data->kdims2, kernel_tmp);
	md_free(kernel_tmp);

	fft(data->N, data->odims, data->flags, dst, dst);
}

static void conv_fft_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(conv_fft_s, _data);

	complex float* kernel_tmp = NULL;
	complex float* output_tmp = NULL;

	output_tmp = md_alloc_sameplace(data->N, data->odims, CFL_SIZE, data->kernel);
	ifft(data->N, data->odims, data->flags,output_tmp, src);

	kernel_tmp = md_alloc_sameplace(data->N, data->kdims2, CFL_SIZE, data->kernel);
	md_ztenmulc(data->N, data->kdims2, kernel_tmp, data->odims, output_tmp, data->idims, data->image);
	md_free(output_tmp);

	fftmod(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);
	fft(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);
	md_resize_center(data->N, data->kdims, dst, data->kdims2, kernel_tmp, CFL_SIZE);
	md_zsmul(data->N, data->kdims, dst, dst, conjf(data->scaling));
}

static void conv_fft_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(conv_fft_s, _data);

	complex float* image_tmp = NULL;
	complex float* kernel_tmp = NULL;

	image_tmp = md_alloc_sameplace(data->N, data->idims, CFL_SIZE, data->kernel);
	ifft(data->N, data->idims, data->flags, image_tmp, src);

	kernel_tmp = md_alloc_sameplace(data->N, data->kdims2, CFL_SIZE, data->kernel);
	md_resize_center(data->N, data->kdims2, kernel_tmp, data->kdims, data->kernel, CFL_SIZE);

	ifft(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);
	ifftmod(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);

	md_ztenmul(data->N, data->odims, dst, data->idims, image_tmp, data->kdims2, kernel_tmp);
	md_free(kernel_tmp);
	md_free(image_tmp);

	fft(data->N, data->odims, data->flags, dst, dst);
}

static void conv_fft_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(conv_fft_s, _data);

	complex float* output_tmp = NULL;
	complex float* kernel_tmp = NULL;

	output_tmp = md_alloc_sameplace(data->N, data->odims, CFL_SIZE, data->kernel);
	ifft(data->N, data->odims, data->flags, output_tmp, src);

	kernel_tmp = md_alloc_sameplace(data->N, data->kdims2, CFL_SIZE, data->kernel);
	md_resize_center(data->N, data->kdims2, kernel_tmp, data->kdims, data->kernel, CFL_SIZE);
	ifft(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);
	ifftmod(data->N, data->kdims2, data->flags, kernel_tmp, kernel_tmp);

	md_ztenmulc(data->N, data->idims, dst, data->odims, output_tmp, data->kdims2, kernel_tmp);
	md_free(kernel_tmp);
	md_free(output_tmp);
	fft(data->N, data->idims, data->flags, dst, dst);
}

static void conv_fft_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(conv_fft_s, _data);

	md_free(data->kernel);
	md_free(data->image);

	xfree(data->odims);
	xfree(data->idims);
	xfree(data->kdims);
	xfree(data->kdims2);

	xfree(data);
}

static struct nlop_s* nlop_conv_fft_cyclic_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N])
{
	complex float scaling = 1.;
	long kdims2[N];

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i)){

			kdims2[i] = idims[i];
			assert(odims[i] == idims[i]);
			assert(odims[i] >= kdims[i]);
			scaling *= (complex float)idims[i];

			if ( idims[i] % 2 == 0)
				scaling *= cpowf(I, (complex float)idims[i]/2.);
			else
				scaling *= cpowf(I, ((complex float)idims[i] - 1.)*((complex float)idims[i] - 1.) / (2. *(complex float)idims[i]));
		} else {

			assert((1 == idims[i]) || (idims[i] == odims[i]) || (idims[i] == kdims[i]));
			assert((1 == kdims[i]) || (kdims[i] == odims[i]) || (kdims[i] == idims[i]));
			assert((1 == odims[i]) || (kdims[i] == odims[i]) || (idims[i] == odims[i]));
			kdims2[i] = kdims[i];
		}

	PTR_ALLOC(struct conv_fft_s, data);
	SET_TYPEID(conv_fft_s, data);

	PTR_ALLOC(long[N], nodims);
	PTR_ALLOC(long[N], nidims);
	PTR_ALLOC(long[N], nkdims);
	PTR_ALLOC(long[N], nkdims2);

	md_copy_dims(N, *nodims, odims);
	md_copy_dims(N, *nidims, idims);
	md_copy_dims(N, *nkdims, kdims);
	md_copy_dims(N, *nkdims2, kdims2);

	data->N = N;
	data->odims = *PTR_PASS(nodims);
	data->idims = *PTR_PASS(nidims);
	data->kdims = *PTR_PASS(nkdims);
	data->kdims2 = *PTR_PASS(nkdims2);

	data->flags=flags;
	data->scaling = 1./scaling;

	// will be initialized later, to transparently support GPU
	data->image = NULL;
	data->kernel = NULL;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], odims);

	long nl_ostrs[1][N];
	md_calc_strides(N, nl_ostrs[0], nl_odims[0], CFL_SIZE);

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], data->idims);
	md_copy_dims(N, nl_idims[1], data->kdims);

	long nl_istrs[2][N];
	md_calc_strides(N, nl_istrs[0], nl_idims[0], CFL_SIZE);
	md_calc_strides(N, nl_istrs[1], nl_idims[1], CFL_SIZE);


	return nlop_generic_create2(1, N, nl_odims, nl_ostrs, 2, N, nl_idims, nl_istrs, CAST_UP(PTR_PASS(data)), conv_fft_fun,
				   (nlop_fun_t[2][1]){ { conv_fft_der1 }, { conv_fft_der2 } },
				   (nlop_fun_t[2][1]){ { conv_fft_adj1 }, { conv_fft_adj2 } }, NULL, NULL, conv_fft_del);
}

struct nlop_s* nlop_convcorr_fft_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum PADDING conv_pad, bool conv)
{
	long odims2[N];
	long idims2[N];
	for (int i = 0; i < N; i++){
		if (MD_IS_SET(flags, i)){

			assert(odims[i] <= idims[i]);
			assert(kdims[i] <= idims[i]);
			idims2[i] = idims[i] + (kdims[i] - 1);
			odims2[i] = odims[i] + (kdims[i] - 1);
		} else {
			assert((1 == idims[i]) || (idims[i] == odims[i]) || (idims[i] == kdims[i]));
			assert((1 == kdims[i]) || (kdims[i] == odims[i]) || (kdims[i] == idims[i]));
			assert((1 == odims[i]) || (kdims[i] == odims[i]) || (idims[i] == odims[i]));
			idims2[i] = idims[i];
			odims2[i] = odims[i];
		}
	}

	struct nlop_s* result = NULL;
	struct nlop_s* niresize;
	struct nlop_s* noresize;
	struct nlop_s* nconv_cyclic;
	struct nlop_s* nchained;
	struct nlop_s* nchained_perm;

	switch (conv_pad){

		case PAD_CYCLIC:

			result = nlop_conv_fft_cyclic_create(N, flags, odims, idims, kdims);
			break;

		case PAD_SAME:

			niresize = nlop_resize_create(N, idims2, idims, false);
			nconv_cyclic = nlop_conv_fft_cyclic_create(N, flags, odims2, idims2, kdims);

			nchained = nlop_chain2(niresize, 0, nconv_cyclic, 0);
			int perm[2]={1,0};
			nchained_perm = nlop_permute_inputs(nchained, 2, perm);
			nlop_free(nconv_cyclic);
			nlop_free(nchained);
			nlop_free(niresize);

			noresize = nlop_resize_create(N, odims, odims2, false);
			nlop_free(noresize);

			result = nlop_chain2(nchained_perm,0,noresize,0);
			nlop_free(nchained_perm);
			nlop_free(noresize);
			break;

		case PAD_VALID:

			for (int i = 0; i < N; i++)
				if(MD_IS_SET(flags, i))
					assert(odims[i] == idims[i] - (kdims[i] - 1));

			nconv_cyclic = nlop_conv_fft_cyclic_create(N, flags, odims2, idims, kdims);
			noresize = nlop_resize_create(N, odims, odims2, true);
			result = nlop_chain2(nconv_cyclic,0,noresize,0);
			nlop_free(noresize);
			break;

		case PAD_CAUSAL:

			assert(0);//not implemented
			break;

		case PAD_SYMMETRIC:

			assert(0);//not implemented
			break;

		case PAD_REFLECT:

			assert(0);//not implemented
			break;
	}

	if (!conv)
		result = nlop_chain2_FF(nlop_from_linop_F(linop_flip_create(N, kdims, flags)), 0, result, 1);

	return result;
}
