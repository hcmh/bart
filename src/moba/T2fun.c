/* Copyright 2018-2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2019 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 */


#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "T2fun.h"



struct T2_s {

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
	complex float* rho;
	complex float* z;

	complex float* tmp_map;
	complex float* tmp_data;
	complex float* tmp_exp;

	complex float* tmp_drho;
	complex float* tmp_dz;

	complex float* TI;

	float scaling_z;
};

DEF_TYPEID(T2_s);

// Calculate Model: rho .*exp(-scaling_z.*z.*TI), TI = [0,1]
static void T2_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// rho
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->rho, data->in_dims, src, CFL_SIZE);

	// z
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->z, data->in_dims, src, CFL_SIZE);

	// -1*scaling_z.*z
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->z, -1 * data->scaling_z);

	// exp(-TI.*scaling_z.*z), TI = [0,1]

	for(int k = 0; k < data->TI_dims[5]; k++)
		md_zsmul2(data->N, data->map_dims, data->out_strs, (void*)data->tmp_exp + data->out_strs[5] * k, data->map_strs, (void*)data->tmp_map, data->TI[k]);

	// md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->map_strs, data->tmp_map, data->TI_strs, data->TI);

	md_zexp(data->N, data->out_dims, data->tmp_exp, data->tmp_exp);

	// Calculating derivatives
	// drho
	md_zsmul(data->N, data->out_dims, data->tmp_drho, data->tmp_exp, 1.0);

	// model:
	// rho.*exp(-TI.*scaling_z.*z), TI = [0,1]
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->rho, data->out_strs, data->tmp_exp);

	long img_dims[data->N];
	md_select_dims(data->N, FFT_FLAGS, img_dims, data->map_dims);

	// dz: z' = -rho.*scaling_z.*TI.*exp(-TI.*scaling_z.*z)
	// TI.*exp(-TI.*scaling_z.*z), TI = [0,1]

	for (int s = 0; s < data->out_dims[13]; s++)
		for(int k = 0; k < data->TI_dims[5]; k++)
			md_zsmul(data->N, img_dims, (void*)data->tmp_exp + data->out_strs[5] * k + data->out_strs[13] * s, (void*)data->tmp_exp + data->out_strs[5] * k + data->out_strs[13] * s, data->TI[k]);

	// md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->out_strs, data->tmp_exp, data->TI_strs, data->TI);
	md_zsmul(data->N, data->out_dims, data->tmp_exp, data->tmp_exp, -1*data->scaling_z);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dz, data->map_strs, data->rho, data->out_strs, data->tmp_exp);
}

static void T2_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// tmp = drho
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);

	// dst = rho' * drho
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_drho);

	// tmp =  dz
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);

	// dst = dst + dz * z'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dz);
}

static void T2_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;


	// conj(rho') * src
	md_zmulc2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, src, data->out_strs, data->tmp_drho);

	// sum (conj(rho') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);

	// dst[0] = sum (conj(rho') * src, t)
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// conj(z').*src
	md_zmulc2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, src, data->out_strs, data->tmp_dz);

	// sum (conj(z') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);

	//md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

	// dst[1] = sum (conj(z') * src, t)
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}

static void T2_del(const nlop_data_t* _data)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);

	md_free(data->rho);
	md_free(data->z);

	md_free(data->TI);

	md_free(data->tmp_map);
	md_free(data->tmp_data);
	md_free(data->tmp_exp);

	md_free(data->tmp_drho);
	md_free(data->tmp_dz);

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


struct nlop_s* nlop_T2_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N], const complex float* TI, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct T2_s, data);
	SET_TYPEID(T2_s, data);


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
	data->rho = my_alloc(N, map_dims, CFL_SIZE);
	data->z = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_map = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_data = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_exp = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_drho = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_dz = my_alloc(N, out_dims, CFL_SIZE);
	data->TI = md_alloc(N, TI_dims, CFL_SIZE);

	md_copy(N, TI_dims, data->TI, TI, CFL_SIZE);

	data->scaling_z = 0.1;

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T2_fun, T2_der, T2_adj, NULL, NULL, T2_del);
}

