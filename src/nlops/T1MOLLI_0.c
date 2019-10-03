/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "T1MOLLI_0.h"


struct zT1relax_s {

	INTERFACE(nlop_data_t);

	int N;
        
	const long* dims;

	const long* strs;
 
	complex float* xn;
        complex float* M0;
        complex float* M_start;
        complex float* dR1;
        complex float* dM0;
        complex float* tmp;
};

DEF_TYPEID(zT1relax_s);

static void zexp_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(zT1relax_s, _data);
        assert(4 == N);

	complex float* dst1 = args[0]; 
	const complex float* src1 = args[1]; // M_start
        const complex float* src2 = args[2]; // M0
        const complex float* src3 = args[3]; // R1
        
        
//         md_zsmul2(data->N, data->dims, data->strs, data->xn, data->dims, src, -1);
        md_zsmul(data->N, data->dims, data->xn, src3, -1.0);
        md_zexp(data->N, data->dims, data->xn, data->xn);
        
        md_copy(data->N, data->dims, data->M_start, src1, CFL_SIZE);
        md_copy(data->N, data->dims, data->M0, src2, CFL_SIZE);
        
        // M_start - M0        
        md_zsub(data->N, data->dims, data->tmp, data->M_start, data->M0);
        
        md_zmul(data->N, data->dims, dst1, data->tmp, data->xn);
        
        md_zadd(data->N, data->dims, dst1, dst1, data->M0);
}

static void zexp_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zT1relax_s, _data);
	md_zmul(data->N, data->dims, dst, src, data->xn);

}

static void zexp_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zT1relax_s, _data);
        md_zfill(data->N, data->dims, data->tmp, 1.0);
        md_zsub(data->N, data->dims, data->dM0, data->tmp, data->xn);
        md_zmul(data->N, data->dims, dst, src, data->dM0);
}

static void zexp_der3(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zT1relax_s, _data);
          // M_start - M0        
        md_zsub(data->N, data->dims, data->tmp, data->M_start, data->M0);
        md_zsmul(data->N, data->dims, data->dR1, data->xn, -1.0);
        md_zmul(data->N, data->dims, data->dR1, data->tmp, data->dR1);

        md_zmul(data->N, data->dims, dst, src, data->dR1);
}

static void zexp_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zT1relax_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void zexp_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zT1relax_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->dM0);
}

static void zexp_adj3(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zT1relax_s, _data);
	md_zmulc(data->N, data->dims, dst, src, data->dR1);
}

static void zexp_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zT1relax_s, _data);

	md_free(data->xn);
        md_free(data->M0);
        md_free(data->dR1);
        md_free(data->M_start);
        md_free(data->dM0);
        md_free(data->tmp);
	xfree(data->dims);
        xfree(data->strs);
	xfree(data);
}


struct nlop_s* nlop_T1relax_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zT1relax_s, data);
	SET_TYPEID(zT1relax_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	
	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = md_alloc(N, dims, CFL_SIZE);
        
        data->M_start = md_alloc(N, dims, CFL_SIZE);
        data->M0 = md_alloc(N, dims, CFL_SIZE);
        data->dR1 = md_alloc(N, dims, CFL_SIZE);
        data->dM0 = md_alloc(N, dims, CFL_SIZE);
        data->tmp = md_alloc(N, dims, CFL_SIZE);
        
        PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, dims, CFL_SIZE);
	data->strs = *PTR_PASS(nostr);
        
        long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->dims);
        
        long nl_ostr[1][N];
	md_copy_strides(N, nl_ostr[0], data->strs);
        
         
        long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], data->dims);
	md_copy_dims(N, nl_idims[1], data->dims);
        md_copy_dims(N, nl_idims[2], data->dims);

	long nl_istr[3][N];
	md_copy_strides(N, nl_istr[0], data->strs);
	md_copy_strides(N, nl_istr[1], data->strs);
        md_copy_strides(N, nl_istr[2], data->strs);
        
        
        return nlop_generic_create2(1, N, nl_odims, nl_ostr, 3, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)),
		zexp_fun, (nlop_fun_t[3][1]){ { zexp_der1 }, { zexp_der2 }, { zexp_der3 } }, (nlop_fun_t[3][1]){ { zexp_adj1 }, { zexp_adj2 }, { zexp_adj3 }}, NULL, NULL, zexp_del);
        

}


