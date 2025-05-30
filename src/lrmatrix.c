/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Frank Ong
 * 2015-2016 Martin Uecker
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "linops/linop.h"
#include "linops/sum.h"
#include "linops/someops.h"

#include "iter/iter.h"
#include "iter/lsqr.h"
#include "iter/thresh.h"
#include "iter/prox2.h"

#include "lowrank/lrthresh.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

struct s_data {

	operator_data_t super;

	long size;
};

static DEF_TYPEID(s_data);

// x = (z1 + z2)/2

static void sum_xupdate(const operator_data_t* _data, float /*rho*/, complex float* dst, const complex float* src)
{
	const struct s_data* data = CAST_DOWN(s_data, _data);

	for(int i = 0; i < data->size; i++)
		dst[i] = src[i] / 2.;
}

static void sum_xupdate_free(const operator_data_t* data)
{
	xfree(CAST_DOWN(s_data, data));
}



static const char help_str[] =
		"Perform (multi-scale) low rank matrix completion";



int main_lrmatrix(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	int maxiter = 100;
	float rho = 0.25;
	int blkskip = 2;
	bool randshift = true;
	unsigned long mflags = 1;
	unsigned long flags = ~0UL;
	const char* sum_str = NULL;
	bool noise = false;
        bool decom = false;

	bool llr = false;
	int llrblk = -1;
	bool ls = false;
	bool hogwild = false;
	bool fast = true;
	int initblk = 1;
	int remove_mean = 0;


	const struct opt_s opts[] = {

		OPT_SET('d', &decom, "perform decomposition instead, ie fully sampled"),
		// FIXME: 'd' fell through to u in original version ??!?
		OPT_INT('i', &maxiter, "iter", "maximum iterations."),
		OPT_ULONG('m', &mflags, "flags", "which dimensions are reshaped to matrix columns."),
		OPT_ULONG('f', &flags, "flags", "which dimensions to perform multi-scale partition."),
		OPT_PINT('j', &blkskip, "scale", "block size scaling from one scale to the next one."),
		OPT_PINT('k', &initblk, "size", "smallest block size"),
		OPT_SET('N', &noise, "add noise scale to account for Gaussian noise."),
		OPT_SET('s', &ls, "perform low rank + sparse matrix completion."),
		OPT_PINT('l', &llrblk, "size", "perform locally low rank soft thresholding with specified block size."),
		OPT_OUTFILE('o', &sum_str, "out2", "summed over all non-noise scales to create a denoised output."),
		OPT_SELECT('u', int, &remove_mean, 1, "()"),
		OPT_SELECT('v', int, &remove_mean, 2, "()"),
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_FLOAT('p', &rho, "", "(rho)"),
		OPT_CLEAR('n', &randshift, "(no randshift)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (-1 != llrblk)
		llr = true;


	long idims[DIMS];
	long odims[DIMS];

	// Load input
	complex float* idata = load_cfl(in_file, DIMS, idims);

	// Get levels and block dimensions
	long blkdims[MAX_LEV][DIMS];
	int levels;

	if (llr)
		levels = llr_blkdims(blkdims, flags, idims, llrblk);
	else if (ls)
		levels = ls_blkdims(blkdims, idims);
	else
		levels = multilr_blkdims(blkdims, flags, idims, blkskip, initblk);

	if (noise)
		add_lrnoiseblk(&levels, blkdims, idims);

	debug_printf(DP_INFO, "Number of levels: %d\n", levels);

	// Get outdims
	md_copy_dims(DIMS, odims, idims);
	odims[LEVEL_DIM] = levels;

	complex float* odata = create_cfl(out_file, DIMS, odims);

	md_clear(DIMS, odims, odata, sizeof(complex float));

	// Get pattern
	complex float* pattern = NULL;

        if (!decom) {

                pattern = md_alloc(DIMS, idims, CFL_SIZE);
                estimate_pattern(DIMS, idims, TIME_FLAG, pattern, idata);
        }

	// Initialize algorithm
	iter_conf* iconf;

	struct iter_admm_conf mmconf;
	memcpy(&mmconf, &iter_admm_defaults, sizeof(struct iter_admm_conf));
	mmconf.maxiter = maxiter;
	mmconf.rho = rho;
	mmconf.hogwild = hogwild;
	mmconf.fast = fast;

	iconf = CAST_UP(&mmconf);


	// Initialize operators

	const struct linop_s* sum_op = linop_scaled_sum_create(DIMS, odims, LEVEL_FLAG);
	const struct linop_s* sampling_op = NULL;

        if (!decom) {

                sampling_op = linop_cdiag_create(DIMS, idims, ~0UL, pattern);
                sum_op = linop_chain_FF(sum_op, sampling_op);
        }

	const struct operator_p_s* sum_prox = prox_lineq_create(sum_op, idata);
	const struct operator_p_s* lr_prox = lrthresh_create(odims, randshift, mflags, (const long (*)[])blkdims, 1., noise, remove_mean, false);

	// put into iter2 format
	int num_funs = 2;
	const struct linop_s* eye_op = linop_identity_create(DIMS, odims);
	const struct linop_s* ops[2] = { eye_op, eye_op };
	const struct operator_p_s* prox_ops[2] = { sum_prox, lr_prox };
	long size = 2 * md_calc_size(DIMS, odims);

	struct s_data* s_data = xmalloc(sizeof *s_data);

	*s_data = (struct s_data){ { &TYPEID(s_data) }, size / 2 };

	const struct operator_p_s* sum_xupdate_op = operator_p_create(DIMS, odims, DIMS, odims, CAST_UP(s_data), sum_xupdate, sum_xupdate_free);


	// do recon

	iter2_admm( iconf,
		    NULL,
		    num_funs,
		    prox_ops,
		    ops,
		    NULL,
		    sum_xupdate_op,
		    size, (float*)odata, NULL,
		    NULL);


	// Sum
	if (sum_str) {

		complex float* sdata = create_cfl(sum_str, DIMS, idims);

		long istrs[DIMS];
		long ostrs[DIMS];

		md_calc_strides(DIMS, istrs, idims, sizeof(complex float));
		md_calc_strides(DIMS, ostrs, odims, sizeof(complex float));

		md_clear(DIMS, idims, sdata, sizeof(complex float));

		odims[LEVEL_DIM]--;
		md_zaxpy2(DIMS, odims, istrs, sdata, 1. / sqrt((double)levels), ostrs, odata);
		odims[LEVEL_DIM]++;

		unmap_cfl(DIMS, idims, sdata);
	}


	// Clean up
	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(DIMS, odims, odata);

	linop_free(sum_op);
	linop_free(eye_op);

	operator_p_free(sum_xupdate_op);
	operator_p_free(sum_prox);
	operator_p_free(lr_prox);

	md_free(pattern);

	double end_time = timestamp();
	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

	return 0;
}

