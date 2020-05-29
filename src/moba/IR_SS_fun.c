/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "IR_SS_fun.h"

// #define general  //FIXME: more general, but less gpu efficient

struct IR_SS_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* map_dims;
	const long* TI_dims;
	const long* in_dims;
	const long* out_dims;

	const long* map_strs;
	const long* TI_strs;
	const long* in_strs;
	const long* out_strs;

	// Parameter maps
	complex float* Mss;
	complex float* R1s;

	complex float* tmp_map;
	complex float* tmp_ones;
	complex float* tmp_exp;

	complex float* tmp_dMss;
	complex float* tmp_dR1s;

	complex float* TI;

	float scaling_R1s;
};

DEF_TYPEID(IR_SS_s);

// Calculate Model: Mss - (Mss + Mss) * exp(-t.*R1s), when the start is Mss not M0
static void IR_SS_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct IR_SS_s* data = CAST_DOWN(IR_SS_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// Mss
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->Mss, data->in_dims, src, CFL_SIZE);

	// R1s
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->R1s, data->in_dims, src, CFL_SIZE);

	// -1*scaling_R1s.*R1s
	md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, -1.0*data->scaling_R1s);

	// exp(-t.*scaling_R1s*R1s):

	long img_dims[data->N];
	md_select_dims(data->N, FFT_FLAGS, img_dims, data->map_dims);

#ifdef general
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
#else
	
	for (int s = 0; s < data->out_dims[13]; s++)
		for (int w = 0; w < data->TI_dims[11]; w++)
			for(int k = 0; k < data->TI_dims[5]; k++)
				md_zsmul(data->N, img_dims, (void*)data->tmp_exp + data->out_strs[5] * k + data->out_strs[11] * w + data->out_strs[13] * s,
					(void*)data->tmp_map + data->map_strs[11] * w + data->map_strs[13] * s, data->TI[k + data->TI_dims[5] * w]);
#endif

	md_zexp(data->N, data->out_dims, data->tmp_exp, data->tmp_exp);

	// 2.0 * Mss
	// md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->Mss, 2.0);
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->Mss, 2.0);

	// 2.0 * Mss.*exp(-t.*scaling_R1s*R1s)
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_exp);

	// Mss - 2.0 * Mss..*exp(-t.*scaling_R1s*R1s)
	md_zsub2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Mss, data->out_strs, dst);

	// Calculating derivatives

	// Mss' = 1 - 2.0 * exp(-t.*scaling_R1s.*R1s)
	md_zfill(data->N, data->out_dims, data->tmp_dMss, 1.0);
	md_zaxpy(data->N, data->out_dims, data->tmp_dMss, -2.0, data->tmp_exp);

	// t*exp(-t.*scaling_R1s*R1s):

#ifdef general
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->out_strs, data->tmp_exp, data->TI_strs, data->TI);
#else 

	for (int s = 0; s < data->out_dims[13]; s++)
		for (int w = 0; w < data->TI_dims[11]; w++)
			for(int k = 0; k < data->TI_dims[5]; k++)
				md_zsmul(data->N, img_dims, (void*)data->tmp_exp + data->out_strs[5] * k + data->out_strs[11] * w + data->out_strs[13] * s,
					(void*)data->tmp_exp + data->out_strs[5] * k + data->out_strs[11] * w + data->out_strs[13] * s, data->TI[k + data->TI_dims[5] * w]);
#endif

	// 2.0 * scaling_R1s * t * exp(-t.*scaling_R1s.*R1s)
	md_zsmul(data->N, data->out_dims, data->tmp_exp, data->tmp_exp, 2.0 * data->scaling_R1s);

	// R1s' = 2.0 * scaling_R1s * Mss * t * exp(-t.*scaling_R1s.*R1s)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1s, data->map_strs, data->Mss, data->out_strs, data->tmp_exp);
}

static void IR_SS_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct IR_SS_s* data = CAST_DOWN(IR_SS_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// tmp = dMss
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	//const complex float* tmp_Mss = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	// dst = dMss * Mss'
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dMss);

	// tmp =  dR1s
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	//const complex float* tmp_R1s = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = dst + dR1s * R1s'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dR1s);
}

static void IR_SS_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct IR_SS_s* data = CAST_DOWN(IR_SS_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;


	// sum (conj(Mss') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dMss);

	// dst[0] = sum (conj(Mss') * src, t)
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// sum (conj(R1s') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dR1s);
// 	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

	// dst[1] = sum (conj(R1s') * src, t)
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}

static void IR_SS_del(const nlop_data_t* _data)
{
	struct IR_SS_s* data = CAST_DOWN(IR_SS_s, _data);

	md_free(data->Mss);
	md_free(data->R1s);

	md_free(data->TI);

	md_free(data->tmp_map);
	md_free(data->tmp_ones);
	md_free(data->tmp_exp);

	md_free(data->tmp_dMss);
	md_free(data->tmp_dR1s);

	xfree(data->map_dims);
	xfree(data->TI_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->map_strs);
	xfree(data->TI_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	xfree(data);
}


struct nlop_s* nlop_IR_SS_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N], const complex float* TI, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct IR_SS_s, data);
	SET_TYPEID(IR_SS_s, data);


	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(long[N], ntidims);
	md_copy_dims(N, *ntidims, TI_dims);
	data->TI_dims = *PTR_PASS(ntidims);

	PTR_ALLOC(long[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nmstr);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr);

	PTR_ALLOC(long[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
	data->in_strs = *PTR_PASS(nistr);

	PTR_ALLOC(long[N], ntistr);
	md_calc_strides(N, *ntistr, TI_dims, CFL_SIZE);
	data->TI_strs = *PTR_PASS(ntistr);

	data->N = N;
	data->Mss = my_alloc(N, map_dims, CFL_SIZE);
	data->R1s = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_map = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_ones = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_exp = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_dMss = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_dR1s = my_alloc(N, out_dims, CFL_SIZE);
#ifdef general
	data->TI = my_alloc(N, TI_dims, CFL_SIZE);
#else
	data->TI = md_alloc(N, TI_dims, CFL_SIZE);
#endif
	md_copy(N, TI_dims, data->TI, TI, CFL_SIZE);

	data->scaling_R1s = 1.0;

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), IR_SS_fun, IR_SS_der, IR_SS_adj, NULL, NULL, IR_SS_del);
}
