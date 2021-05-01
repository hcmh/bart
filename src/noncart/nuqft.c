/* Copyright 2016-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	2016-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "linops/linop.h"

#include "nuqft.h"



static void get_coord(unsigned int N, unsigned long flags, float cnst[1], float lin[N], float mom[N][N], long pos[N], const long tdims[N], const long tstrs[N], const complex float* traj)
{
	assert(0 == pos[0]);
	int k = 0;

#if 0
	cnst[0] = -crealf(MD_ACCESS(N, tstrs, pos, traj));

	k++;
#else
	cnst[0] = 0.;
#endif

	for (int i = 0; i < (int)N; i++) {

		lin[i] = 0.;

		if (MD_IS_SET(flags, i)) {

			pos[0] = k;
			lin[i] = -crealf(MD_ACCESS(N, tstrs, pos, traj));
			k++;
		}
	}

	for (int i = 0; i < (int)N; i++) {

		for (int j = 0; j <= i; j++) {

			if (   MD_IS_SET(flags, i)
			    && MD_IS_SET(flags, j)) {

				pos[0] = k;
				mom[i][j] = -crealf(MD_ACCESS(N, tstrs, pos, traj));
				mom[j][i] = mom[i][j];
				k++;

			} else {

				mom[i][j] = mom[j][i] = 0.;
			}
		}
	}

	assert(tdims[0] == k);
	pos[0] = 0;
}



void nuqft_forward2(unsigned int N, unsigned long flags, 
			const long kdims[N], const long kstrs[N], complex float* ksp,
			const long idims[N], const long istrs[N], const complex float* img,
			const long tdims[N], const long tstrs[N], const complex float* traj)
{
	assert(1 == kdims[0]);
	assert(md_check_compat(N, ~0, kdims, tdims));

	long tmp_dims[N];
	long tmp_strs[N];

	md_select_dims(N, flags, tmp_dims, idims);
	md_calc_strides(N, tmp_strs, tmp_dims, CFL_SIZE);

	complex float* tmp = md_alloc(N, tmp_dims, CFL_SIZE);

	long kstrs2[N];
	for (unsigned int i = 0; i < N; i++)
		kstrs2[i] = MD_IS_SET(flags, i) ? 0 : kstrs[i];


	md_clear2(N, kdims, kstrs, ksp, CFL_SIZE);

	long pos[N];
	for (unsigned int i = 0; i < N; i++)
		pos[i] = 0;

	do {
		float cnst[1];
		float lin[N];
		float mom[N][N];

		get_coord(N, flags, cnst, lin, mom, pos, tdims, tstrs, traj);

		quadratic_phase2(N, tmp_dims, cnst[0], lin, mom, tmp);
		md_zfmac2(N, idims, kstrs2, &MD_ACCESS(N, kstrs, pos, ksp), istrs, img, tmp_strs, tmp);

	} while (md_next(N, tdims, ~MD_BIT(0), pos));


	md_free(tmp);
}

/**
 *
 */
void nuqft_adjoint2(unsigned int N, unsigned long flags, 
			const long idims[N], const long istrs[N], complex float* img,
			const long kdims[N], const long kstrs[N], const complex float* ksp,
			const long tdims[N], const long tstrs[N], const complex float* traj)
{
	assert(1 == kdims[0]);
	assert(md_check_compat(N, ~0, kdims, tdims));

	long tmp_dims[N];
	long tmp_strs[N];

	md_select_dims(N, flags, tmp_dims, idims);
	md_calc_strides(N, tmp_strs, tmp_dims, CFL_SIZE);

	complex float* tmp = md_alloc(N, tmp_dims, CFL_SIZE);

	long kstrs2[N];
	for (unsigned int i = 0; i < N; i++)
		kstrs2[i] = MD_IS_SET(flags, i) ? 0 : kstrs[i];


	md_clear2(N, idims, istrs, img, CFL_SIZE);

	long pos[N];
	for (unsigned int i = 0; i < N; i++)
		pos[i] = 0;

	do {
		float cnst[1];
		float lin[N];
		float mom[N][N];

		get_coord(N, flags, cnst, lin, mom, pos, tdims, tstrs, traj);
	
		quadratic_phase2(N, tmp_dims, cnst[0], lin, mom, tmp);
		md_zfmacc2(N, idims, istrs, img, kstrs2, &MD_ACCESS(N, kstrs, pos, ksp), tmp_strs, tmp);

	} while (md_next(N, tdims, ~MD_BIT(0), pos));


	md_free(tmp);
}


void nuqft_forward(unsigned int N, unsigned long flags,
			const long odims[N], complex float* out,
			const long idims[N], const complex float* in,
			const long tdims[N], const complex float* traj)
{
	long ostrs[N];
	long istrs[N];
	long tstrs[N];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, idims, CFL_SIZE);
	md_calc_strides(N, tstrs, tdims, CFL_SIZE);	// FL_SIZE

	nuqft_forward2(N, flags, odims, ostrs, out, idims, istrs, in, tdims, tstrs, traj);
}


struct nuqft_s {

	linop_data_t base;

	unsigned int N;
	unsigned long flags;

	long* kdims;
	long* idims;
	long* tdims;
	long* kstrs;
	long* istrs;
	long* tstrs;

	const complex float* traj;
};


static void nuqft_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct nuqft_s* data = CONTAINER_OF(_data, const struct nuqft_s, base);
	unsigned int N = data->N;

	nuqft_forward2(N, data->flags,
			data->kdims, data->kstrs, out,
			data->idims, data->istrs, in,
			data->tdims, data->tstrs, data->traj);
}

static void nuqft_adj(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct nuqft_s* data = CONTAINER_OF(_data, const struct nuqft_s, base);
	unsigned int N = data->N;

	nuqft_adjoint2(N, data->flags,
			data->idims, data->istrs, out,
			data->kdims, data->kstrs, in,
			data->tdims, data->tstrs, data->traj);
}

static void nuqft_delete(const linop_data_t* _data)
{
	const struct nuqft_s* data = CONTAINER_OF(_data, const struct nuqft_s, base);

	xfree(data->kdims);
	xfree(data->idims);
	xfree(data->tdims);
	xfree(data->kstrs);
	xfree(data->istrs);
	xfree(data->tstrs);

	xfree(data);
}

const struct linop_s* nuqft_create2(unsigned int N, unsigned long flags,
					const long odims[N], const long ostrs[N],
					const long idims[N], const long istrs[N],
					const long tdims[N], const complex float* traj)
{
	PTR_ALLOC(struct nuqft_s, data);

	data->N = N;
	data->flags = flags;
	data->traj = traj;

	data->kdims = *TYPE_ALLOC(long[N]);
	data->kstrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->kdims, odims);
	md_copy_strides(N, data->kstrs, ostrs);

	data->idims = *TYPE_ALLOC(long[N]);
	data->istrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->idims, idims);
	md_copy_strides(N, data->istrs, istrs);

	data->tdims = *TYPE_ALLOC(long[N]);
	data->tstrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->tdims, tdims);
	md_calc_strides(N, data->tstrs, tdims, CFL_SIZE);

	return linop_create2(N, odims, ostrs, N, idims, istrs, &PTR_PASS(data)->base,
			nuqft_apply, nuqft_adj, NULL, NULL, nuqft_delete);
}

const struct linop_s* nuqft_create(unsigned int N, unsigned long flags, const long odims[N], const long idims[N], const long tdims[N], const complex float* traj)
{
	return nuqft_create2(N, flags, odims, MD_STRIDES(N, odims, CFL_SIZE), idims, MD_STRIDES(N, idims, CFL_SIZE), tdims, traj);
}

