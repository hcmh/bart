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

#include "T1relax_so.h"


struct zT1relax_so_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* map_dims;
        const long* out_dims;
        const long* TI_dims;

	const long* map_strs;
        const long* out_strs;
        const long* TI_strs;

	complex float* xn;
        complex float* M0;
        complex float* M_start;
        complex float* dR1;
        complex float* dM0;
        complex float* tmp;

        complex float* TI;
};

DEF_TYPEID(zT1relax_so_s);

static void zT1relax_so_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(zT1relax_so_s, _data);
        assert(4 == N);

	complex float* dst1 = args[0];
	const complex float* src1 = args[1]; // M_start
        const complex float* src2 = args[2]; // M0
        const complex float* src3 = args[3]; // R1


        // exp(-t.*R1)
        md_zsmul(data->N, data->map_dims, data->tmp, src3, -1.0);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->xn, data->map_strs, data->tmp, data->TI_strs, data->TI);

        md_zexp(data->N, data->out_dims, data->xn, data->xn);

        md_copy(data->N, data->map_dims, data->M_start, src1, CFL_SIZE);
        md_copy(data->N, data->map_dims, data->M0, src2, CFL_SIZE);

        // M_start - M0
        md_zsub(data->N, data->map_dims, data->tmp, data->M_start, data->M0);

        // (M_start - M0 )*exp(-t*R1)
        md_zmul2(data->N, data->out_dims, data->out_strs, data->dM0, data->map_strs, data->tmp, data->out_strs, data->xn);

        // M0 + (M_start - M0 )*exp(-t*R1)
        md_zadd2(data->N, data->out_dims, data->out_strs, dst1, data->map_strs, data->M0, data->out_strs, data->dM0);

        // derivatives (first output)
	// dM_start: data->xn
        // dM0
        md_zfill(data->N, data->map_dims, data->tmp, 1.0);
        md_zsub2(data->N, data->out_dims, data->out_strs, data->dM0, data->map_strs, data->tmp, data->out_strs, data->xn);

        // dR1
        // M_start - M0
        md_zsub(data->N, data->map_dims, data->tmp, data->M_start, data->M0);

        // -t*(M_start - M0)*exp(-t*R1)
        md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1, data->map_strs, data->tmp, data->out_strs, data->xn);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1, data->out_strs, data->dR1, data->TI_strs, data->TI);
        md_zsmul(data->N, data->out_dims, data->dR1, data->dR1, -1.0);
}

static void zT1relax_so_der_0_0(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1relax_so_s, _data);

        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, src, data->out_strs, data->xn);
}


static void zT1relax_so_der_0_1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1relax_so_s, _data);
        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, src, data->out_strs, data->dM0);
}

static void zT1relax_so_der_0_2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1relax_so_s, _data);
        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, src, data->out_strs, data->dR1);
}


static void zT1relax_so_adj_0_0(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1relax_so_s, _data);

        // sum (conj(M_start') * src, t)
	md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, dst, data->out_strs, src, data->out_strs, data->xn);
}

static void zT1relax_so_adj_0_1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1relax_so_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, dst, data->out_strs, src, data->out_strs, data->dM0);
}

static void zT1relax_so_adj_0_2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zT1relax_so_s, _data);
        md_clear(data->N, data->map_dims, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, dst, data->out_strs, src, data->out_strs, data->dR1);
}


static void zT1relax_so_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zT1relax_so_s, _data);

	md_free(data->xn);
        md_free(data->M0);
        md_free(data->dR1);
        md_free(data->M_start);
        md_free(data->dM0);
        md_free(data->tmp);

        md_free(data->TI);

	xfree(data->map_dims);
        xfree(data->map_strs);

        xfree(data->out_dims);
        xfree(data->out_strs);

        xfree(data->TI_dims);
        xfree(data->TI_strs);
	xfree(data);
}


struct nlop_s* nlop_T1relax_so_create(int N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], const complex float* TI, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

        PTR_ALLOC(struct zT1relax_so_s, data);
	SET_TYPEID(zT1relax_so_s, data);

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

	data->xn = my_alloc(N, out_dims, CFL_SIZE);

        data->M_start = my_alloc(N, map_dims, CFL_SIZE);
        data->M0 = my_alloc(N, map_dims, CFL_SIZE);
        data->dR1 = my_alloc(N, out_dims, CFL_SIZE);
        data->dM0 = my_alloc(N, out_dims, CFL_SIZE);
        data->tmp = my_alloc(N, map_dims, CFL_SIZE);

        PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nostr);

        PTR_ALLOC(long[N], nostr1);
	md_calc_strides(N, *nostr1, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr1);

        PTR_ALLOC(long[N], nostr2);
	md_calc_strides(N, *nostr2, TI_dims, CFL_SIZE);
	data->TI_strs = *PTR_PASS(nostr2);

        long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->out_dims);

        long nl_ostr[1][N];
	md_copy_strides(N, nl_ostr[0], data->out_strs);

	data->TI = my_alloc(N, TI_dims, CFL_SIZE);

        md_copy(N, TI_dims, data->TI, TI, CFL_SIZE);

        long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], data->map_dims);
	md_copy_dims(N, nl_idims[1], data->map_dims);
        md_copy_dims(N, nl_idims[2], data->map_dims);

	long nl_istr[3][N];
	md_copy_strides(N, nl_istr[0], data->map_strs);
	md_copy_strides(N, nl_istr[1], data->map_strs);
        md_copy_strides(N, nl_istr[2], data->map_strs);


        return nlop_generic_create2(1, N, nl_odims, nl_ostr, 3, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)), zT1relax_so_fun,
                                    (nlop_der_fun_t[3][1]){{ zT1relax_so_der_0_0 }, {zT1relax_so_der_0_1 }, { zT1relax_so_der_0_2 }},
                                    (nlop_der_fun_t[3][1]){{ zT1relax_so_adj_0_0 }, {zT1relax_so_adj_0_1 }, { zT1relax_so_adj_0_2 }},
                                    NULL, NULL, zT1relax_so_del);


}