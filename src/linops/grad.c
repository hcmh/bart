/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>
#include <strings.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "grad.h"



static void grad_dims(unsigned int D, long dims2[D], int d, unsigned int flags, const long dims[D])
{
	md_copy_dims(D, dims2, dims);

	assert(1 == dims[d]);
	assert(!MD_IS_SET(flags, d));

	dims2[d] = bitcount(flags);
}



static void grad_op(unsigned int D, const long dims[D], int d, unsigned int flags, unsigned long order,
		    const enum BOUNDARY_CONDITION bc, bool reverse, complex float *out, const complex float *in)
{
	assert((1 == order) || (2 == order));
	unsigned int N = bitcount(flags);
	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	long strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	unsigned int flags2 = flags;

	for (unsigned int i = 0; i < N; i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);
		long ooffset = i * strs[d];
		if (1 == order)
			(reverse ? md_zfdiff_backwards2 : md_zfdiff2)(D, dims1, lsb, bc, strs, (void *)out + ooffset, strs1, in);

		if (2 == order) {
			md_zfdiff_central2(D, dims1, lsb, bc, reverse, strs, (void *)out + ooffset, strs1, in);
			md_zsmul2(D, dims1, strs, (void *)out + ooffset, strs, (void *)out + ooffset, .5);

		}
	}
	assert(0 == flags2);
}



static void grad_adjoint(unsigned int D, const long dims[D], int d, unsigned int flags, unsigned long order,
			 const enum BOUNDARY_CONDITION bc, bool reverse, complex float *out, const complex float *in)
{
	assert((1 == order) || (2 == order));
	unsigned int N = bitcount(flags);
	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	long strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	unsigned int flags2 = flags;

	complex float *tmp = md_alloc_sameplace(D, dims1, CFL_SIZE, out);

	md_clear(D, dims1, out, CFL_SIZE);
	md_clear(D, dims1, tmp, CFL_SIZE);

	for (unsigned int i = 0; i < N; i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		if (1 == order) {
			(reverse ? md_zfdiff2 : md_zfdiff_backwards2)(D, dims1, lsb, bc, strs1, tmp, strs,
								      (void *)in + i * strs[d]);
			//Special Case: For "SAME", zfdiff & zfdiff_backwards are not adjoint
			//'at the boundaries'
			if (BC_SAME == bc) {
				long pos[D];
				md_select_dims(D, ~MD_BIT(lsb), pos, dims1);

				long ioff = i * strs[d] + (reverse ? 0 : 1) * strs[lsb];
				long ooff = 0;
				md_zsmul2(D, pos, strs1, (void *)tmp + ooff, strs, (void *)in + ioff, reverse ? 1. : -1.);

				ioff = i * strs[d] + (dims1[lsb] - (reverse ? 2 : 1)) * strs[lsb];
				ooff = (dims1[lsb] - 1) * strs1[lsb];
				md_zsmul2(D, pos, strs1, (void *)tmp + ooff, strs, (void *)in + ioff, reverse ? -1 : 1.);
			}
		}
		if (2 == order) {
			md_zfdiff_central2(D, dims1, lsb, bc, !reverse, strs1, tmp, strs, (void *)in + i * strs[d]);

			if (BC_SAME == bc) {
				long pos[D];
				md_select_dims(D, ~MD_BIT(lsb), pos, dims1);

				long ioff = i * strs[d], ooff = 0;
				md_zaxpy2(D, pos, strs1, (void *)tmp + ooff, reverse ? 2. : -2., strs, (void *)in + ioff);

				ioff = i * strs[d] + (dims1[lsb] - 1) * strs[lsb];
				ooff = (dims1[lsb] - 1) * strs1[lsb];
				md_zaxpy2(D, pos, strs1, (void *)tmp + ooff, reverse ? -2. : 2., strs, (void *)in + ioff);
			}

			md_zsmul2(D, dims1, strs1, tmp, strs1, tmp, .5);
		}

		md_zadd(D, dims1, out, out, tmp);
	}

	md_free(tmp);

	assert(0 == flags2);
}



struct grad_s {

	INTERFACE(linop_data_t);

	int N;
	int d;
	long *dims;
	unsigned long flags;
	unsigned int order;
	bool reverse;
	enum BOUNDARY_CONDITION bc;
};

static DEF_TYPEID(grad_s);

static void grad_op_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_op(data->N, data->dims, data->d, data->flags, data->order, data->bc, data->reverse, dst, src);
}

static void grad_op_adjoint(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_adjoint(data->N, data->dims, data->d, data->flags, data->order, data->bc, data->reverse, dst, src);
}

static void grad_op_normal(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	complex float *tmp = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	// this could be implemented more efficiently
	grad_op(data->N, data->dims, data->d, data->flags, data->order, data->bc, data->reverse, tmp, src);
	grad_adjoint(data->N, data->dims, data->d, data->flags, data->order, data->bc, data->reverse, dst, tmp);

	md_free(tmp);
}

static void grad_op_free(const linop_data_t *_data)
{
	const auto data = CAST_DOWN(grad_s, _data);

	xfree(data->dims);
	xfree(data);
}

struct linop_s *linop_fd_create(long N, const long dims[N], int d, unsigned int flags, unsigned int order,
				const enum BOUNDARY_CONDITION bc, bool reverse)
{
	assert((1 == order) || (2 == order));

	PTR_ALLOC(struct grad_s, data);
	SET_TYPEID(grad_s, data);

	int NO = N;

	if (N == d) {

		// as a special case, id d is one after the last dimensions,
		// we extend the output dimensions by one.

		NO++;

	} else {

		assert(1 == dims[d]);
	}

	long dims2[NO];
	md_copy_dims(N, dims2, dims);
	dims2[d] = 1;

	grad_dims(NO, dims2, d, flags, dims2);

	data->N = NO;
	data->d = d;
	data->flags = flags;

	data->dims = *TYPE_ALLOC(long[N + 1]);

	data->order = order;
	data->reverse = reverse;
	data->bc = bc;

	md_copy_dims(NO, data->dims, dims2);

	return linop_create(NO, dims2, N, dims, CAST_UP(PTR_PASS(data)), grad_op_apply, grad_op_adjoint, grad_op_normal, NULL, grad_op_free);
}

struct linop_s *linop_grad_create(long N, const long dims[N], int d, unsigned int flags)
{
	return linop_fd_create(N, dims, d, flags, 1, BC_PERIODIC, false);
}

struct linop_s *linop_div_create(long N, const long dims[N], int d, unsigned int flags, const unsigned int order, const enum BOUNDARY_CONDITION bc)
{
	PTR_ALLOC(struct linop_s, op2);

	assert(dims[d] == bitcount(flags));
	long gdims[N];
	md_select_dims(N, ~MD_BIT(d), gdims, dims);

	auto op = linop_fd_create(N, gdims, d, flags, order, bc, true);
	op2 = (struct linop_s*)linop_get_adjoint(op); //FIXME: we should make linops consistently const
	linop_free(op);

	return PTR_PASS(op2);
}
