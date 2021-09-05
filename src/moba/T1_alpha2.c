/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 */


#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "T1_alpha2.h"


struct zT1_alpha2_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* map_dims;
        const long* out_dims;
        const long* TI_dims;

	const long* map_strs;
        const long* out_strs;
        const long* TI_strs;

	complex float* xn;
        complex float* xn_tmp;
        complex float* M0;
        complex float* alpha;
        complex float* R1;
        complex float* M_start;
        complex float* tmp;
        complex float* R1s;

        complex float* dalpha;
        complex float* dM0;
        complex float* dR1;

        complex float* dM_start_1;
        complex float* dM0_1;
        complex float* dR1_1;
        complex float* dalpha_1;

        complex float* TI;

        float scaling_alpha;
        float regularization;


};

DEF_TYPEID(zT1_alpha2_s);

static void zT1_alpha2_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        assert(6 == N);

	complex float* dst1 = args[0];
        complex float* dst2 = args[1];
	const complex float* src1 = args[2]; // M_start
        const complex float* src2 = args[3]; // M0
        const complex float* src3 = args[4]; // R1
        const complex float* src4 = args[5]; // alpha

        md_copy(data->N, data->map_dims, data->M_start, src1, CFL_SIZE);
        md_copy(data->N, data->map_dims, data->M0, src2, CFL_SIZE);
        md_copy(data->N, data->map_dims, data->R1, src3, CFL_SIZE);
        md_copy(data->N, data->map_dims, data->alpha, src4, CFL_SIZE);

        // R1s = R1 + alpha * scaling_alpha
	md_zsmul(data->N, data->map_dims, data->R1s, data->alpha, data->scaling_alpha);
	md_zadd(data->N, data->map_dims, data->R1s, data->R1, data->R1s);

        md_zsmul(data->N, data->map_dims, data->tmp, data->R1s, -1.);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->xn, data->map_strs, data->tmp, data->TI_strs, data->TI);

        // for(int k = 0; k < (data->TI_dims[5]); k++)
	// 	md_zsmul2(data->N, data->map_dims, data->out_strs, (void*)data->xn + data->out_strs[5] * k, data->map_strs, (void*)data->R1s, -1.*data->TI[k]);

        md_zexp(data->N, data->out_dims, data->xn, data->xn);

        //M0 * R1/R1s - M_start
	md_zdiv_reg(data->N, data->map_dims, data->dM0_1, data->R1, data->R1s, data->regularization); // dM0_1 =  R1 / R1s
        md_zmul(data->N, data->map_dims, data->tmp, data->dM0_1, data->M0);

        md_zsub(data->N, data->map_dims, data->dR1_1, data->tmp, data->M_start);

	// (M0 * R1/(R1 + alpha) - M_start).*exp(-t.*scaling_alpha*alpha)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dalpha, data->map_strs, data->dR1_1, data->out_strs, data->xn);

	//Model: (M0 * R1/(R1 + alpha) -(-M_start + M0 * R1/(R1 + alpha)).*exp(-t.*scaling_alpha*alpha))
	md_zsub2(data->N, data->out_dims, data->out_strs, dst1, data->map_strs, data->tmp, data->out_strs, data->dalpha);

        long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

        pos[TE_DIM] = data->TI_dims[TE_DIM]-1;
	md_copy_block(data->N, pos, data->map_dims, dst2, data->out_dims, dst1, CFL_SIZE);

        // derivatives (first output)
        // dR1 = 1 - exp(-t*R1s)
        // dM0
        md_zfill(data->N, data->map_dims, data->tmp, 1.0);
        md_zsub2(data->N, data->out_dims, data->out_strs, data->dR1, data->map_strs, data->tmp, data->out_strs, data->xn); // 1 - exp(-t*R1s)
        md_zmul2(data->N, data->out_dims, data->out_strs, data->dM0, data->out_strs, data->dR1, data->map_strs, data->dM0_1);

        //dalpha
        //R1/(scaling_alpha*alpha.^2) * M0
        md_zmul(data->N, data->map_dims, data->tmp, data->R1s, data->R1s);
	md_zdiv_reg(data->N, data->map_dims, data->tmp, data->R1, data->tmp, data->regularization);
        md_zmul(data->N, data->map_dims, data->tmp, data->M0, data->tmp);

        // (exp(-t*R1s) - 1)
        // md_zsmul(data->N, data->out_dims, data->dR1, data->dR1, -1.);
        // md_zfill(data->N, data->map_dims, data->dalpha_1, 1.0);
	// md_zsub2(data->N, data->out_dims, data->out_strs, data->dR1, data->out_strs, data->xn, data->map_strs, data->dalpha_1);

        md_zmul2(data->N, data->out_dims, data->out_strs, data->dalpha, data->map_strs, data->tmp, data->out_strs, data->dR1);

        
        md_zmul2(data->N, data->out_dims, data->out_strs, data->xn_tmp, data->out_strs, data->xn, data->TI_strs, data->TI);
        // md_zsmul(data->N, data->out_dims, data->xn_tmp, data->xn_tmp, data->scaling_alpha);

        // (M0*R1/(scaling_alpha*alpha) - M_start) * t * exp(-t * R1s)
        md_zmul2(data->N, data->out_dims, data->out_strs, data->xn_tmp, data->map_strs, data->dR1_1, data->out_strs, data->xn_tmp);

        md_zsub(data->N, data->out_dims, data->dalpha, data->xn_tmp, data->dalpha);

        // (M0 / R1s)  * (1 - exp(-t * R1s))
        md_zdiv_reg(data->N, data->map_dims, data->tmp, data->M0, data->R1s, data->regularization);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->xn_tmp, data->map_strs, data->tmp, data->out_strs, data->dR1);

        md_zadd(data->N, data->out_dims, data->dR1, data->dalpha, data->xn_tmp);

        // alpha' * scaling_alpha
        md_zsmul(data->N, data->out_dims, data->dalpha, data->dalpha, data->scaling_alpha);

        md_copy_block(data->N, pos, data->map_dims, data->dM_start_1, data->out_dims, data->xn, CFL_SIZE);
        md_copy_block(data->N, pos, data->map_dims, data->dM0_1, data->out_dims, data->dM0, CFL_SIZE);
        md_copy_block(data->N, pos, data->map_dims, data->dR1_1, data->out_dims, data->dR1, CFL_SIZE);
        md_copy_block(data->N, pos, data->map_dims, data->dalpha_1, data->out_dims, data->dalpha, CFL_SIZE);
}

static void zT1_alpha2_der_0_0(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);

        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, src, data->out_strs, data->xn);
}


static void zT1_alpha2_der_0_1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, src, data->out_strs, data->dM0);
}

static void zT1_alpha2_der_0_2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, src, data->out_strs, data->dR1);
}

static void zT1_alpha2_der_0_3(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, src, data->out_strs, data->dalpha);
}

static void zT1_alpha2_der_1_0(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);

        md_zmul(data->N, data->map_dims, dst, src, data->dM_start_1);
}


static void zT1_alpha2_der_1_1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);

        md_zmul(data->N, data->map_dims, dst, src, data->dM0_1);
}

static void zT1_alpha2_der_1_2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);

        md_zmul(data->N, data->map_dims, dst, src, data->dR1_1);
}

static void zT1_alpha2_der_1_3(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);

        md_zmul(data->N, data->map_dims, dst, src, data->dalpha_1);
}

static void zT1_alpha2_adj_0_0(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
// 	md_zmulc(data->N, data->out_dims, dst, src, data->xn);

        	// sum (conj(M_start') * src, t)
	md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, dst, data->out_strs, src, data->out_strs, data->xn);
}

static void zT1_alpha2_adj_0_1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, dst, data->out_strs, src, data->out_strs, data->dM0);
}

static void zT1_alpha2_adj_0_2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, dst, data->out_strs, src, data->out_strs, data->dR1);
}

static void zT1_alpha2_adj_0_3(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, dst, data->out_strs, src, data->out_strs, data->dalpha);
}

static void zT1_alpha2_adj_1_0(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);

	md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc(data->N, data->map_dims, dst, src, data->dM_start_1);
}

static void zT1_alpha2_adj_1_1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc(data->N, data->map_dims, dst, src, data->dM0_1);
}

static void zT1_alpha2_adj_1_2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc(data->N, data->map_dims, dst, src, data->dR1_1);
}

static void zT1_alpha2_adj_1_3(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1_alpha2_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc(data->N, data->map_dims, dst, src, data->dalpha_1);
}

static void zT1_alpha2_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zT1_alpha2_s, _data);

	md_free(data->xn);
        md_free(data->xn_tmp);
        md_free(data->M0);
        md_free(data->R1);
        md_free(data->alpha);
        md_free(data->R1s);
        md_free(data->M_start);

        md_free(data->dR1);
        md_free(data->dM0);
        md_free(data->dalpha);

        md_free(data->tmp);

        md_free(data->dR1_1);
        md_free(data->dM_start_1);
        md_free(data->dM0_1);
        md_free(data->dalpha_1);


        md_free(data->TI);

	xfree(data->map_dims);
        xfree(data->map_strs);

        xfree(data->out_dims);
        xfree(data->out_strs);

        xfree(data->TI_dims);
        xfree(data->TI_strs);
	xfree(data);
}


struct nlop_s* nlop_T1_alpha2_create(int N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], const complex float* TI, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct zT1_alpha2_s, data);
	SET_TYPEID(zT1_alpha2_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
        data->map_dims = *PTR_PASS(ndims);

        PTR_ALLOC(long[N], ndims1);
	md_copy_dims(N, *ndims1, out_dims);
        data->out_dims = *PTR_PASS(ndims1);

        PTR_ALLOC(long[N], ndims2);
	md_copy_dims(N, *ndims2, TI_dims);
        data->TI_dims = *PTR_PASS(ndims2);

	data->N = N;

        data->M_start = my_alloc(N, map_dims, CFL_SIZE);
        data->M0 = my_alloc(N, map_dims, CFL_SIZE);
        data->R1 = my_alloc(N, map_dims, CFL_SIZE);
        data->R1s = my_alloc(N, map_dims, CFL_SIZE);
        data->alpha = my_alloc(N, map_dims, CFL_SIZE);
        data->tmp = my_alloc(N, map_dims, CFL_SIZE);

        data->dR1 = my_alloc(N, out_dims, CFL_SIZE);
        data->dM0 = my_alloc(N, out_dims, CFL_SIZE);
        data->dalpha = my_alloc(N, out_dims, CFL_SIZE);
        data->xn = my_alloc(N, out_dims, CFL_SIZE);
        data->xn_tmp = my_alloc(N, out_dims, CFL_SIZE);


        data->dM_start_1 = my_alloc(N, map_dims, CFL_SIZE);
        data->dM0_1 = my_alloc(N, map_dims, CFL_SIZE);
        data->dR1_1 = my_alloc(N, map_dims, CFL_SIZE);
        data->dalpha_1 = my_alloc(N, map_dims, CFL_SIZE);

        PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nostr);

        PTR_ALLOC(long[N], nostr1);
	md_calc_strides(N, *nostr1, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr1);

        PTR_ALLOC(long[N], nostr2);
	md_calc_strides(N, *nostr2, TI_dims, CFL_SIZE);
	data->TI_strs = *PTR_PASS(nostr2);

        long nl_odims[2][N];
	md_copy_dims(N, nl_odims[0], data->out_dims);
        md_copy_dims(N, nl_odims[1], data->map_dims);

        long nl_ostr[2][N];
	md_copy_strides(N, nl_ostr[0], data->out_strs);
        md_copy_strides(N, nl_ostr[1], data->map_strs);

	data->TI = my_alloc(N, TI_dims, CFL_SIZE);

        data->scaling_alpha = 0.2;
        data->regularization = 1e-6;

        md_copy(N, TI_dims, data->TI, TI, CFL_SIZE);

        long nl_idims[4][N];
	md_copy_dims(N, nl_idims[0], data->map_dims);
	md_copy_dims(N, nl_idims[1], data->map_dims);
        md_copy_dims(N, nl_idims[2], data->map_dims);
        md_copy_dims(N, nl_idims[3], data->map_dims);

	long nl_istr[4][N];
	md_copy_strides(N, nl_istr[0], data->map_strs);
	md_copy_strides(N, nl_istr[1], data->map_strs);
        md_copy_strides(N, nl_istr[2], data->map_strs);
        md_copy_strides(N, nl_istr[3], data->map_strs);


        return nlop_generic_create2(2, N, nl_odims, nl_ostr, 4, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)), zT1_alpha2_fun,
                                    (nlop_der_fun_t[4][2]){{ zT1_alpha2_der_0_0, zT1_alpha2_der_1_0 }, {zT1_alpha2_der_0_1, zT1_alpha2_der_1_1 }, { zT1_alpha2_der_0_2,  zT1_alpha2_der_1_2 }, { zT1_alpha2_der_0_3,  zT1_alpha2_der_1_3 }},
                                    (nlop_der_fun_t[4][2]){{ zT1_alpha2_adj_0_0, zT1_alpha2_adj_1_0 }, {zT1_alpha2_adj_0_1, zT1_alpha2_adj_1_1 }, { zT1_alpha2_adj_0_2 , zT1_alpha2_adj_1_2 }, { zT1_alpha2_adj_0_3 , zT1_alpha2_adj_1_3 }}, NULL, NULL, zT1_alpha2_del);




}