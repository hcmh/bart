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

static const struct nlop_s* nlop_fftc_create(long N, const long dims[N], unsigned int flags, bool inv)
{
	auto lfft = (inv ? linop_ifftc_create : linop_fftc_create)(N, dims, flags);
	auto nfft = nlop_from_linop(lfft);
	linop_free(lfft);
	return nfft;
}

struct nlop_s* nlop_conv_create(long N, unsigned int flags, const long odims[N], const long idims1[N], const long idims2[N])
{
	auto nl = nlop_tenmul_create(N, odims, idims1, idims2);

	auto ffto = nlop_fftc_create(N, odims, flags, true);
	auto nl2 = nlop_chain(nl, ffto);
	nlop_free(ffto);
	nlop_free(nl);

	auto ffti1 = nlop_fftc_create(N, idims1, flags, false);
	auto nl3 = nlop_chain2(ffti1, 0, nl2, 0);
	nlop_free(ffti1);
	nlop_free(nl2);

	auto ffti2 = nlop_fftc_create(N, idims2, flags, false);
	auto nl4 = nlop_chain2(ffti2, 0, nl3, 1);
	nlop_free(ffti2);
	nlop_free(nl3);

	return nl4;
}

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

	const struct iovec_s* idom;
	const struct iovec_s* kdom;
	const struct iovec_s* codom;

	unsigned int flags;

	bool conv;

	complex float* k;
	complex float* i;
};

DEF_TYPEID(convcorr_geom_s);


static void convcorr_geom_initialize(struct convcorr_geom_s* data, const complex float* arg)
{
	if (NULL == data->k)
		data->k = md_alloc_sameplace(data->N, data->kdom->dims, CFL_SIZE, arg);

	if (NULL == data->i)
		data->i = md_alloc_sameplace(data->N, data->idom->dims, CFL_SIZE, arg);
}


static void convcorr_geom_fun(const nlop_data_t* _data, int N, complex float* args[N])
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
	convcorr_geom_initialize(data, dst);

	md_zconj2(data->N, data->idom->dims, MD_STRIDES(data->N, data->idom->dims, CFL_SIZE), data->i, data->idom->strs, src1);
	md_zconj2(data->N, data->kdom->dims, MD_STRIDES(data->N, data->kdom->dims, CFL_SIZE), data->k, data->kdom->strs, src2);

	long mdims[2 * data->N];
	long ostrs2[2 * data->N];
	long kstrs2[2 * data->N];
	long istrs2[2 * data->N];

	const complex float *krn = src2;
	krn += calc_convcorr_geom(data->N, data->flags, mdims, ostrs2, kstrs2, istrs2, data->codom->dims, data->codom->strs, data->kdom->dims, data->kdom->strs, data->idom->dims, data->idom->strs, data->conv) / CFL_SIZE;
#if 0
	//This needs other ordering in calc_convcor_geom
	long out_dims[data->N];
	md_select_dims(data->N, md_nontriv_strides(data->N, ostrs2), out_dims, mdims);
	long tmp_out_dims[data->N];
	md_select_dims(data->N, md_nontriv_strides(data->N, ostrs2), tmp_out_dims, mdims);

	for (int i = 0; i < data->N; i++)
		if (MD_IS_SET(data->flags, i)){

			tmp_out_dims[i] = data->idom->dims[i];
			mdims[i] = data->idom->dims[i];
		}

	md_calc_strides(data->N, ostrs2, tmp_out_dims, CFL_SIZE);

	long new_insize[] = {1};
	for (int i = 0; i < 2 * data->N; i ++)
		new_insize[0] += (mdims[i]-1) * istrs2[i] / CFL_SIZE;
	long old_insize[] = {md_calc_size(data->N, data->idom->dims)};

	complex float* tmpin = md_alloc_sameplace(1, new_insize, CFL_SIZE, src1);
	md_resize(1, new_insize, tmpin, old_insize, src1, CFL_SIZE);

	complex float* tmpout = md_alloc_sameplace(data->N, tmp_out_dims, CFL_SIZE, src1);
	md_clear(data->N, tmp_out_dims, tmpout, CFL_SIZE);

	md_zfmac2(2 * data->N, mdims, ostrs2, tmpout, istrs2, tmpin, kstrs2, krn);
	md_resize(data->N, out_dims, dst, tmp_out_dims, tmpout, CFL_SIZE);

	md_free(tmpin);
	md_free(tmpout);

#else
	md_ztenmul2(2 * data->N, mdims, ostrs2, dst, istrs2, src1, kstrs2, krn);
#endif
	PRINT_TIMER("convgeos");
}


static void convcorr_geom_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	long mdims[2 * data->N];
	long ostrs2[2 * data->N];
	long kstrs2[2 * data->N];
	long istrs2[2 * data->N];

	const complex float* krn = src;
	const complex float* in = data->i;

	krn += calc_convcorr_geom(data->N, data->flags, mdims, ostrs2, kstrs2, istrs2, data->codom->dims, data->codom->strs, data->kdom->dims, data->kdom->strs, data->idom->dims, MD_STRIDES(data->N, data->idom->dims, CFL_SIZE), data->conv) / CFL_SIZE;

	md_ztenmulc2(2 * data->N, mdims, ostrs2, dst, kstrs2, krn, istrs2, in);

	PRINT_TIMER("convgeo der2s");
}

static void convcorr_geom_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);
	long mdims[2 * data->N];
	long ostrs2[2 * data->N];
	long kstrs2[2 * data->N];
	long istrs2[2 * data->N];

	complex float *krn = dst;
	krn += calc_convcorr_geom(data->N, data->flags, mdims, ostrs2, kstrs2, istrs2, data->codom->dims, data->codom->strs, data->kdom->dims, MD_STRIDES(data->N, data->kdom->dims,CFL_SIZE), data->idom->dims, data->idom->strs, data->conv) / CFL_SIZE;

#if 0
	long out_dims[data->N];
	md_select_dims(data->N, md_nontriv_strides(data->N, ostrs2), out_dims, mdims);

	long tmp_out_dims[data->N];
	md_select_dims(data->N, md_nontriv_strides(data->N, ostrs2), tmp_out_dims, mdims);

	for (int i = 0; i < data->N; i++)
		if (MD_IS_SET(data->flags, i)){

			tmp_out_dims[i] = data->idom->dims[i];
			mdims[i] = data->idom->dims[i];
		}

	md_calc_strides(data->N, ostrs2, tmp_out_dims, CFL_SIZE);

	long new_insize[] = {1};
	for (int i = 0; i < 2 * data->N; i ++)
		new_insize[0] += (mdims[i]-1) * istrs2[i] / CFL_SIZE;
	long old_insize[] = {md_calc_size(data->N, data->idom->dims)};

	complex float* tmpin = md_alloc_sameplace(1, new_insize, CFL_SIZE, data->i);
	md_resize(1, new_insize, tmpin, old_insize, data->i, CFL_SIZE);

	complex float* tmpout = md_alloc_sameplace(data->N, tmp_out_dims, CFL_SIZE, src);
	md_resize(data->N, tmp_out_dims, tmpout, out_dims, src, CFL_SIZE);

	md_clear(data->N, data->kdom->dims, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, mdims, kstrs2, krn, ostrs2, tmpout, istrs2, tmpin);

	md_free(tmpin);
	md_free(tmpout);
#else
	md_ztenmul2(2 * data->N, mdims, kstrs2, krn, ostrs2, src, istrs2, data->i);
#endif
	PRINT_TIMER("convgeo adj2s");
}

static void convcorr_geom_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	long mdims[2 * data->N];
	long ostrs2[2 * data->N];
	long kstrs2[2 * data->N];
	long istrs2[2 * data->N];

	const complex float* krn = data->k;
	const complex float* in = src;

	krn += calc_convcorr_geom(data->N, data->flags, mdims, ostrs2, kstrs2, istrs2, data->codom->dims, data->codom->strs,  data->kdom->dims, MD_STRIDES(data->N, data->kdom->dims, CFL_SIZE), data->idom->dims, data->idom->strs, data->conv) / CFL_SIZE;

	md_ztenmulc2(2 * data->N, mdims, ostrs2, dst, istrs2, in, kstrs2, krn);

	PRINT_TIMER("convgeo der1s");
}

static void convcorr_geom_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	long mdims[2 * data->N];
	long ostrs2[2 * data->N];
	long kstrs2[2 * data->N];
	long istrs2[2 * data->N];

	const complex float* krn = data->k;

	krn += calc_convcorr_geom(data->N, data->flags, mdims, ostrs2, kstrs2, istrs2,
					data->codom->dims, data->codom->strs,
					data->kdom->dims, MD_STRIDES(data->N, data->kdom->dims, CFL_SIZE),
					data->idom->dims, data->idom->strs,
					data->conv) / CFL_SIZE;
#if 0
	long out_dims[data->N];
	md_select_dims(data->N, md_nontriv_strides(data->N, ostrs2), out_dims, mdims);

	long tmp_out_dims[data->N];
	md_select_dims(data->N, md_nontriv_strides(data->N, ostrs2), tmp_out_dims, mdims);

	for (int i = 0; i < data->N; i++)
		if (MD_IS_SET(data->flags, i)){

			tmp_out_dims[i] = data->idom->dims[i];
			mdims[i] = data->idom->dims[i];
		}

	md_calc_strides(data->N, ostrs2, tmp_out_dims, CFL_SIZE);

	long new_insize[] = {1};
	for (int i = 0; i < 2 * data->N; i ++)
		new_insize[0] += (mdims[i]-1) * istrs2[i] / CFL_SIZE;
	long old_insize[] = {md_calc_size(data->N, data->idom->dims)};

	complex float* tmpin = md_alloc_sameplace(1, new_insize, CFL_SIZE, data->i);
	complex float* tmpout = md_alloc_sameplace(data->N, tmp_out_dims, CFL_SIZE, src);
	md_resize(data->N, tmp_out_dims, tmpout, out_dims, src, CFL_SIZE);

	md_clear(data->N, data->idom->dims, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, mdims, istrs2, tmpin, ostrs2, tmpout, kstrs2, krn);
	md_resize(1, old_insize, dst, new_insize, tmpin, CFL_SIZE);
	md_free(tmpin);
	md_free(tmpout);
#else
	md_ztenmul2(2 * data->N, mdims, istrs2, dst, ostrs2, src, kstrs2, krn);
#endif
	PRINT_TIMER("convgeo adj1s");
}


static void convcorr_geom_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	md_free(data->i);
	md_free(data->k);

	iovec_free(data->idom);
	iovec_free(data->kdom);
	iovec_free(data->codom);

	xfree(data);
}

static struct nlop_s* nlop_convcorr_geom_create2(long N, unsigned int flags, const long odims[N], const long ostr[N], const long idims[N], const long istr[N], const long kdims[N], const long kstr[N], bool conv)
{
	for (int i = 0; i < N; i++)
		if MD_IS_SET(flags, i)
			assert(odims[i] == idims[i] - (kdims[i] - 1));

	PTR_ALLOC(struct convcorr_geom_s, data);
	SET_TYPEID(convcorr_geom_s, data);

	data->conv = conv;
	data->flags = flags;

	// will be initialized later, to transparently support GPU
	data->i = NULL;
	data->k = NULL;

	long nl_odims[1][N];
	md_select_dims(N, md_nontriv_strides(N, ostr), nl_odims[0], odims);

	long nl_ostr[1][N];
	md_copy_strides(N, nl_ostr[0], ostr);

	long nl_idims[2][N];
	md_select_dims(N, md_nontriv_strides(N, istr), nl_idims[0], idims);
	md_select_dims(N, md_nontriv_strides(N, kstr), nl_idims[1], kdims);

	long nl_istr[2][N];
	md_copy_strides(N, nl_istr[0], istr);
	md_copy_strides(N, nl_istr[1], kstr);

	data->N = N;
	data->idom = iovec_create2(N, nl_idims[0], istr, CFL_SIZE);
	data->kdom = iovec_create2(N, nl_idims[1], kstr, CFL_SIZE);
	data->codom = iovec_create2(N, nl_odims[0], ostr, CFL_SIZE);

	return nlop_generic_create2(1, N, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)), convcorr_geom_fun,
				    (nlop_fun_t[2][1]){ { convcorr_geom_der1 }, { convcorr_geom_der2 } },
				    (nlop_fun_t[2][1]){ { convcorr_geom_adj1 }, { convcorr_geom_adj2 } }, NULL, NULL, convcorr_geom_del);
}

static struct nlop_s* nlop_convcorr_geom_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], bool conv)
{
	return nlop_convcorr_geom_create2(N, flags, odims, MD_STRIDES(N, odims, CFL_SIZE), idims, MD_STRIDES(N, idims, CFL_SIZE), kdims, MD_STRIDES(N, kdims, CFL_SIZE), conv);
}

struct nlop_s* nlop_conv_geom_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum CONV_PAD conv_pad)
{
	struct nlop_s* result = NULL;
	long idims2[N];

	switch (conv_pad){
		case PADDING_VALID:

			result = nlop_convcorr_geom_create(N, flags, odims, idims, kdims, true);
			break;

	case PADDING_SAME:

		for (int i = 0; i < N; i++){
			if (MD_IS_SET(flags, i))
				idims2[i] = idims[i] + kdims[i] - 1;
			else
				idims2[i] = idims[i];
		}

		struct nlop_s* nresize = nlop_resize_create(N, idims2, idims, true);
		struct nlop_s* nconv_valid = nlop_convcorr_geom_create(N, flags, odims, idims2, kdims, true);
		struct nlop_s* nchained = nlop_chain2(nresize, 0, nconv_valid, 0);
		int perm[2]={1,0};
		result = nlop_permute_inputs(nchained, 2, perm);

		nlop_free(nresize);
		nlop_free(nconv_valid);
		nlop_free(nchained);

		break;

	case PADDING_CYCLIC:

		assert(false);//not implemented
		break;

	case PADDING_CAUSAL:

		assert(false);//not implemented
		break;
	}

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

struct nlop_s* nlop_conv_fft_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum CONV_PAD conv_pad)
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

		case PADDING_CYCLIC:

			result = nlop_conv_fft_cyclic_create(N, flags, odims, idims, kdims);
			break;

		case PADDING_SAME:

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

		case PADDING_VALID:

			for (int i = 0; i < N; i++)
				if(MD_IS_SET(flags, i))
					assert(odims[i] == idims[i] - (kdims[i] - 1));

			nconv_cyclic = nlop_conv_fft_cyclic_create(N, flags, odims2, idims, kdims);
			noresize = nlop_resize_create(N, odims, odims2, true);
			result = nlop_chain2(nconv_cyclic,0,noresize,0);
			nlop_free(noresize);
			break;

		case PADDING_CAUSAL:

			assert(0);//not implemented
			break;
	}

	return result;
}

