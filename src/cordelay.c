/* Copyright 2017-2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2019 Sebastian Rosenzweig
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/qform.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"






static const char usage_str[] = "<traj> <k> <output> ";
static const char help_str[] = "Gradient Delay correction on data\n";

int main_cordelay(int argc, char* argv[])
{
	float gdelays[3] = { 0., 0., 0. };
	const char* b0_file = NULL;

	const struct opt_s opts[] = {

		OPT_FLVEC3('q', &gdelays, "delays", "gradient delays: x, y, xy"),
		OPT_STRING('B', &b0_file, "B0", "B0 correction file"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


	num_init();

	long k_dims[DIMS];
	complex float* k = load_cfl(argv[2], DIMS, k_dims);

	assert((k_dims[TIME_DIM] == 1) && (k_dims[SLICE_DIM] == 1));


	long traj_dims[DIMS];
	complex float* traj = load_cfl(argv[1], DIMS, traj_dims);

	md_check_compat(DIMS, ~(READ_FLAG|COIL_FLAG), k_dims, traj_dims);

	// Calculate projection angle
	long traj_red_dims[DIMS];
	md_select_dims(DIMS, ~PHS1_FLAG, traj_red_dims, traj_dims);

	complex float* traj_red = md_alloc(DIMS, traj_red_dims, CFL_SIZE);

	long pos[DIMS] = {0};
	md_slice(DIMS, PHS1_FLAG, pos, traj_dims, traj_red, traj, CFL_SIZE);

	long angles_dims[DIMS];
	md_select_dims(DIMS, PHS2_FLAG, angles_dims, k_dims);

	complex float* angles = md_alloc(DIMS, angles_dims, CFL_SIZE);

	long N = k_dims[PHS2_DIM];

	for (int i = 0; i < N; i++)
		angles[i] = M_PI + atan2f(crealf(traj_red[3 * i + 0]), crealf(traj_red[3 * i + 1]));

	complex float* k_cor = create_cfl(argv[3], DIMS, k_dims);

	if (NULL != b0_file) { // 0th order gradient delay correction

		long b0_dims[2];
		complex float* b0 = load_cfl(b0_file, 2, b0_dims);

		assert(b0_dims[1] == k_dims[COIL_DIM]);


		long ph_dims[DIMS];
		md_select_dims(DIMS, PHS2_FLAG, ph_dims, k_dims);

		complex float* ph = md_alloc(DIMS, ph_dims, CFL_SIZE);


		long ph_full_dims[DIMS];
		md_select_dims(DIMS, PHS2_FLAG|COIL_FLAG, ph_full_dims, k_dims);

		complex float* ph_full = md_alloc(DIMS, ph_full_dims, CFL_SIZE);


		long pos[DIMS] = { 0 };

		for (int i = 0; i < k_dims[COIL_DIM]; i++) {

			for (int j = 0; j < N; j++)
				ph[j] = cexpf(-1.i * (crealf(b0[i * 2]) * cosf(crealf(angles[j])) + crealf(b0[i * 2 + 1]) * sinf(crealf(angles[j]))));

			pos[COIL_DIM] = i;
			md_copy_block(DIMS, pos, ph_full_dims, ph_full, ph_dims, ph, CFL_SIZE);
		}

		long k_strs[DIMS];
		md_calc_strides(DIMS, k_strs, k_dims, CFL_SIZE);

		long ph_full_strs[DIMS];
		md_calc_strides(DIMS, ph_full_strs, ph_full_dims, CFL_SIZE);

		dump_cfl("ph_new", DIMS, ph_full_dims, ph_full);	// FIXME

		md_zmul2(DIMS, k_dims, k_strs, k_cor, k_strs, k, ph_full_strs, ph_full);

		unmap_cfl(2, b0_dims, b0);

		md_free(ph);
		md_free(ph_full);

	} else { // 1st order gradient delay correction

		long dk_dims[DIMS];
		md_copy_dims(DIMS, dk_dims, angles_dims);

		complex float* dk = md_alloc(DIMS, dk_dims, CFL_SIZE);

		long dk_strs[DIMS];
		md_calc_strides(DIMS, dk_strs, dk_dims, CFL_SIZE);

		// Calculate shifts 'dk'
		for (long i = 0; i < N; i++) 
			dk[i] = quadratic_form(gdelays, angles[i]);


		// Create phase-ramps
		long RO = k_dims[PHS1_DIM];

		complex float* ramps = md_alloc(DIMS, k_dims, CFL_SIZE);

		long ramps_strs[DIMS];
		md_calc_strides(DIMS, ramps_strs, k_dims, CFL_SIZE);


		long ramps_singleton_dims[DIMS];
		md_select_dims(DIMS, PHS1_FLAG, ramps_singleton_dims, k_dims);

		long ramps_singleton_strs[DIMS];
		md_calc_strides(DIMS, ramps_singleton_strs, ramps_singleton_dims, CFL_SIZE);

		complex float* ramps_singleton = md_alloc(DIMS, ramps_singleton_dims, CFL_SIZE);

		for (long i = 0; i < RO; i++)
			ramps_singleton[i] = -RO / 2. + i;


		md_copy2(DIMS, k_dims, ramps_strs, ramps, ramps_singleton_strs, ramps_singleton, CFL_SIZE);

		md_zmul2(DIMS, k_dims, ramps_strs, ramps, ramps_strs, ramps, dk_strs, dk);
		md_zsmul(DIMS, k_dims, ramps, ramps, 2. * M_PI / RO);
		md_zexpj(DIMS, k_dims, ramps, ramps);



		// Shift k-space samples using Fourier-Shift-Theorem
		complex float* k_trans = md_alloc(DIMS, k_dims, CFL_SIZE);

		ifftuc(DIMS, k_dims, PHS1_FLAG, k_trans, k);
		md_zmul(DIMS, k_dims, k_trans, k_trans, ramps);
		fftuc(DIMS, k_dims, PHS1_FLAG, k_cor, k_trans);

		md_free(traj_red);
		md_free(angles);
		md_free(dk);
		md_free(ramps_singleton);
		md_free(ramps);
		md_free(k_trans);
	}

	unmap_cfl(DIMS, traj_dims, traj);
	unmap_cfl(DIMS, k_dims, k);
	unmap_cfl(DIMS, k_dims, k_cor);

	exit(0);
}

