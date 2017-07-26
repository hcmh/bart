/* Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/init.h"

#include "linops/linop.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/lsqr.h"

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "noncart/nufft.h"

#include "sense/model.h"


#ifndef DIMS
#define DIMS 16
#endif


static const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps, const long ksp_dims[DIMS], const long traj_dims[DIMS], const complex float* traj, struct nufft_conf_s conf, const complex float* weights)
{
	long coilim_dims[DIMS];
	long img_dims[DIMS];

	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

	const struct linop_s* fft_op = nufft_create(DIMS, ksp_dims, coilim_dims, traj_dims, traj, weights, conf);
	const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims, img_dims, maps);
	const struct linop_s* lop = linop_chain(maps_op, fft_op);

	linop_free(maps_op);
	linop_free(fft_op);

	return lop;
}


static const char usage_str[] = "<traj> <kspace> <sens> <ref> <output>";
static const char help_str[] = "Real-time SENSE.\n";

	
int main_ncsense(int argc, char* argv[])
{
	struct nufft_conf_s nuconf = nufft_conf_defaults;
	nuconf.toeplitz = false;

	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
	struct lsqr_conf conf = lsqr_defaults;

	const struct opt_s opts[] = {

		OPT_UINT('i', &cgconf.maxiter, "iter", "iterations"),
		OPT_FLOAT('r', &cgconf.l2lambda, "lambda", "regularization"),
	};

	cmdline(&argc, argv, 5, 5, usage_str, help_str, ARRAY_SIZE(opts), opts);


	num_init();


	long max_dims[DIMS];
	long map_dims[DIMS];
	long img_dims[DIMS];
	long coilim_dims[DIMS];
	long ksp_dims[DIMS];
	long traj_dims[DIMS];


	// load kspace and maps and get dimensions

	complex float* kspace_in = load_cfl(argv[2], DIMS, ksp_dims);
	complex float* maps = load_cfl(argv[3], DIMS, map_dims);

	unsigned int map_flags = FFT_FLAGS | SENS_FLAGS;
	for (unsigned int d = 0; d < DIMS; d++)
		if (map_dims[d] > 1)
			map_flags = MD_SET(map_flags, d);


	complex float* traj = load_cfl(argv[1], DIMS, traj_dims);


	md_copy_dims(DIMS, max_dims, ksp_dims);
	md_copy_dims(5, max_dims, map_dims);

	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);

	if (!md_check_compat(DIMS, ~(MD_BIT(MAPS_DIM)|FFT_FLAGS), img_dims, map_dims))
		error("Dimensions of image and sensitivities do not match!\n");

	assert(1 == ksp_dims[MAPS_DIM]);



	long idims[DIMS];
	complex float* ref_data = load_cfl(argv[4], DIMS, idims);

	complex float* out_data = create_cfl(argv[5], DIMS, idims);
	md_clear(DIMS, idims, out_data, CFL_SIZE);


	const struct linop_s* forward_op = sense_nc_init(max_dims, map_dims, maps, ksp_dims, traj_dims, traj, nuconf, NULL);

	complex float* kspace = md_alloc(DIMS, ksp_dims, CFL_SIZE);
	linop_forward(forward_op, DIMS, ksp_dims, kspace, DIMS, idims, ref_data);

	md_zsub(DIMS, ksp_dims, kspace, kspace_in, kspace);


	const struct operator_s* op = lsqr2_create(&conf,
				      iter2_conjgrad, CAST_UP(&cgconf),
				      NULL,
				      forward_op,
				      NULL,
					0, NULL, NULL, NULL);

	operator_apply(op, DIMS, idims, out_data, DIMS, ksp_dims, kspace);

	operator_free(op);

	md_zadd(DIMS, idims, out_data, out_data, ref_data);


	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, traj_dims, traj);
	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, idims, ref_data);
	unmap_cfl(DIMS, idims, out_data);
	exit(0);
}


