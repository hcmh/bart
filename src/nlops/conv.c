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

static void convcorr_geom_set_opts(const nlop_data_t* _data, const struct op_options_s* opts)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if(op_options_is_set_io(opts, 0, 0, OP_APP_CLEAR_DER)){

		md_free(data->src2);
		data->src2 = NULL;
	}
	if(op_options_is_set_io(opts, 0, 1, OP_APP_CLEAR_DER)){

		md_free(data->src1);
		data->src1 = NULL;
	}
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

	bool der1 = !op_options_is_set_io(_data->options, 0, 0, OP_APP_NO_DER);
	bool der2 = !op_options_is_set_io(_data->options, 0, 1, OP_APP_NO_DER);

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


static void convcorr_geom_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if (NULL == data->src1)
		error("Convcorr %x derivative not available\n", data);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmacc2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs2, src + data->shift, data->istrs1, data->src1);

	PRINT_TIMER("der2 convgeo");
}

static void convcorr_geom_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if (NULL == data->src1)
		error("Convcorr %x derivative not available\n", data);

	md_clear(data->N, data->idims2, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, data->mdims, data->istrs2, dst + data->shift, data->ostrs, src, data->istrs1, data->src1);

	PRINT_TIMER("adj2 convgeo");
}

static void convcorr_geom_der1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	START_TIMER;

	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	if (NULL == data->src2)
		error("Convcorr %x derivative not available\n", data);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmacc2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs1, src, data->istrs2, data->src2 + data->shift);

	PRINT_TIMER("der1 convgeo");
}

static void convcorr_geom_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

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

static struct nlop_s* nlop_convcorr_geom_valid_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N],
							bool conv, const long strides[N], const long dilations[N], bool transp)
{
	for (int i = 0; i < N; i++)
		if MD_IS_SET(flags, i)
			assert(idims[i] == strides[i] * (odims[i] - 1) + 1 + (kdims[i] - 1) * dilations[i]);

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
		data->shift = calc_convcorr_geom_strs_dil(N, flags,
							*nmdims, *nistrs1, *nistrs2, *nostrs,
							odims, MD_STRIDES(N, odims, CFL_SIZE),
							kdims, MD_STRIDES(N, kdims, CFL_SIZE),
							idims, MD_STRIDES(N, idims, CFL_SIZE),
							dilations, strides, conv, false) / CFL_SIZE;
	else
		data->shift = calc_convcorr_geom_strs_dil(N, flags,
							*nmdims, *nostrs, *nistrs2, *nistrs1,
							odims, MD_STRIDES(N, odims, CFL_SIZE),
							kdims, MD_STRIDES(N, kdims, CFL_SIZE),
							idims, MD_STRIDES(N, idims, CFL_SIZE),
							dilations, strides, conv, false) / CFL_SIZE;

	data->odims = *PTR_PASS(nodims);
	data->idims1 = *PTR_PASS(nidims1);
	data->idims2 = *PTR_PASS(nidims2);

	data->mdims = *PTR_PASS(nmdims);
	data->ostrs = *PTR_PASS(nostrs);
	data->istrs1 = *PTR_PASS(nistrs1);
	data->istrs2 = *PTR_PASS(nistrs2);

	operator_property_flags_t props[2][1] = {{MD_BIT(OP_PROP_C_LIN)}, {MD_BIT(OP_PROP_C_LIN)}};

	return nlop_generic_with_props_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), convcorr_geom_fun,
				    (nlop_der_fun_t[2][1]){ { convcorr_geom_der1 }, { convcorr_geom_der2 } },
				    (nlop_der_fun_t[2][1]){ { convcorr_geom_adj1 }, { convcorr_geom_adj2 } }, NULL, NULL, convcorr_geom_del, convcorr_geom_set_opts, props, NULL);
}


struct nlop_s* nlop_convcorr_geom_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N],
					enum PADDING conv_pad, bool conv, const long strides[N], const long dilations[N], char transpc)
{
	long ones[N];
	for (int i = 0; i < N; i++)
		ones[i] = 1.;
	if (NULL == strides)
		strides = ones;
	if (NULL == dilations)
		dilations = ones;

	struct nlop_s* result = NULL;

	assert(('N' == transpc) || ('T' == transpc) || ('C' == transpc));

	bool transp = ('N' != transpc);

	if (PAD_VALID == conv_pad) {

		result = nlop_convcorr_geom_valid_create(N, flags, odims, idims, kdims, conv, strides, dilations, transp);
	} else {

		long pad[N];
		long nidims[N];
		for (int i = 0; i < N; i++) {

				if(MD_IS_SET(flags, i))
					pad[i] = kdims[i] / 2 * dilations[i];

				else
					pad[i] = 0;

			nidims[i] = idims[i] + 2 * pad[i];
		}

		result = nlop_convcorr_geom_valid_create(N, flags, odims, nidims, kdims, conv, strides, dilations, transp);

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
