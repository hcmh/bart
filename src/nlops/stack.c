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

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "stack.h"


struct stack_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* idims1;
	const long* idims2;
	const long* odims;

	const long* istrs;
	const long* ostrs;

	long offset;
};

DEF_TYPEID(stack_s);


static void stack_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(stack_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif
	
	md_copy2(data->N, data->idims1, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst, MD_STRIDES(data->N, data->idims1, CFL_SIZE), src1, CFL_SIZE);
	md_copy2(data->N, data->idims2, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst + data->offset, MD_STRIDES(data->N, data->idims2, CFL_SIZE), src2, CFL_SIZE);
}

static void stack_der2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stack_s, _data);
	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_copy2(data->N, data->idims2, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst + data->offset, MD_STRIDES(data->N, data->idims2, CFL_SIZE), src, CFL_SIZE);
}

static void stack_adj2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stack_s, _data);
	md_copy2(data->N, data->idims2, MD_STRIDES(data->N, data->idims2, CFL_SIZE) , dst, MD_STRIDES(data->N, data->odims, CFL_SIZE), src + data->offset, CFL_SIZE);
}

static void stack_der1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{

	const auto data = CAST_DOWN(stack_s, _data);
	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_copy2(data->N, data->idims1, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst, MD_STRIDES(data->N, data->idims1, CFL_SIZE), src, CFL_SIZE);
}

static void stack_adj1(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stack_s, _data);
	md_copy2(data->N, data->idims1, MD_STRIDES(data->N, data->idims1, CFL_SIZE), dst, MD_STRIDES(data->N, data->odims, CFL_SIZE), src, CFL_SIZE);
}


static void stack_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(stack_s, _data);

	xfree(data->idims1);
	xfree(data->idims2);
	xfree(data->odims);

	xfree(data->istrs);
	xfree(data->ostrs);
	
	xfree(data);
}


struct nlop_s* nlop_stack_create(int N, const long odims[N], const long idims1[N], const long idims2[N], unsigned long stack_dim)
{
	

	PTR_ALLOC(struct stack_s, data);
	SET_TYPEID(stack_s, data);

	data->offset = 1;

	for(unsigned int i = 0; i < (unsigned)N; i++)
	{
		if (i == stack_dim)
			assert(odims[i] == (idims1[i] + idims2[i]));
		else
			assert((odims[i] == idims1[i]) && (odims[i] == idims2[i]));

		if (i <= stack_dim)
			data->offset *= idims1[i]; 
			
	}

	PTR_ALLOC(long[N], nodims);
	PTR_ALLOC(long[N], nidims1);
	PTR_ALLOC(long[N], nidims2);
	md_copy_dims(N, *nodims, odims);
	md_copy_dims(N, *nidims1, idims1);
	md_copy_dims(N, *nidims2, idims2);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, idims1, CFL_SIZE);
	data->istrs = *PTR_PASS(nostr);
        
        PTR_ALLOC(long[N], nostr1);
	md_calc_strides(N, *nostr1, odims, CFL_SIZE);
	data->ostrs = *PTR_PASS(nostr1);
	
	data->N = N;
	data->odims = *PTR_PASS(nodims);
	data->idims1 = *PTR_PASS(nidims1);
	data->idims2 = *PTR_PASS(nidims2);
	
	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->odims);

	long nl_ostr[1][N];
	md_copy_strides(N, nl_ostr[0], data->ostrs);
	
	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], data->idims1);
	md_copy_dims(N, nl_idims[1], data->idims2);

	long nl_istr[2][N];
	md_copy_strides(N, nl_istr[0], data->istrs);
	md_copy_strides(N, nl_istr[1], data->istrs);

	return nlop_generic_create2(1, N, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)),
		stack_fun, (nlop_fun_t[2][1]){ { stack_der1 }, { stack_der2 } }, (nlop_fun_t[2][1]){ { stack_adj1 }, { stack_adj2 } }, NULL, NULL, stack_del);
}
