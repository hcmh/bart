/* Copyright 2017-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 *
 *
 * Kai Tobias Block and Martin Uecker, Simple Method for Adaptive
 * Gradient-Delay Compensation in Radial MRI, Annual Meeting ISMRM,
 * Montreal 2011, In Proc. Intl. Soc. Mag. Reson. Med 19: 2816 (2011)
 *
 * Amir Moussavi, Markus Untenberger, Martin Uecker, and Jens Frahm,
 * Correction of gradient-induced phase errors in radial MRI,
 * Magnetic Resonance in Medicine, 71:308-312 (2014)
 *
 * Sebastian Rosenzweig, Hans Christian Holme, Martin Uecker,
 * Simple Auto-Calibrated Gradient Delay Estimation From Few Spokes Using Radial
 * Intersections (RING), Magnetic Resonance in Medicine 81:1898-1906 (2019)
 */

#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/init.h"
#include "num/qform.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "calib/delays.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif





static const char usage_str[] = "<trajectory> <data> [<b0> or <qf>]";
static const char help_str[] = "Estimate gradient delays from radial data.";


int main_estdelay(int argc, char* argv[])
{
	bool do_ring = false;
	struct ring_conf conf = ring_defaults;
	bool do_b0 = false;
	bool do_ac_adaptive = false;
	unsigned int pad_factor = 100;
	bool is_DC = true;

	const struct opt_s opts[] = {

		OPT_SET('R', &do_ring, "RING method"),
		OPT_UINT('p', &conf.pad_factor, "p", "[RING] Padding"),
		OPT_UINT('n', &conf.no_intersec_sp, "n", "[RING] Number of intersecting spokes"),
		OPT_FLOAT('r', &conf.size, "r", "[RING] Central region size"),
		OPT_SET('b', &conf.b0, "[RING] Assume B0 eddy currents"),
		OPT_SET('B', &do_b0, "B0 correction"),
	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (do_ring && do_b0)
		error("Do either b0 or RING");
	
	if (!(do_b0 || do_ring))
		do_ac_adaptive = true;
	
		
	if (0 != pad_factor % 2)
		error("Pad_factor -p should be even\n");


	long tdims[DIMS];
	const complex float* traj = load_cfl(argv[1], DIMS, tdims);

	long tdims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc(DIMS, tdims1, CFL_SIZE);
	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ 0 }, tdims, traj1, traj, CFL_SIZE);

	int N = tdims[2];

	float angles[N];
	for (int i = 0; i < N; i++)
		angles[i] = M_PI + atan2f(crealf(traj1[3 * i + 0]), crealf(traj1[3 * i + 1]));



	// Check if DC component is sampled
	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ [1] = tdims[1] / 2 }, tdims, traj1, traj, CFL_SIZE);
	for (int i = 0; i < N; i++)
		if (0. != cabsf(traj1[3 * i]))
			is_DC = false;
			

	if (do_ring) {

		assert(0 == tdims[1] % 2);
		conf.is_DC = is_DC;
	}


	md_free(traj1);


	long full_dims[DIMS];
	const complex float* full_in = load_cfl(argv[2], DIMS, full_dims);

	// Remove not needed dimensions
	long dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|PHS1_FLAG|PHS2_FLAG|COIL_FLAG, dims, full_dims);

	complex float* in = md_alloc(DIMS, dims, CFL_SIZE);

	long pos[DIMS] = { 0 };
	md_copy_block(DIMS, pos, dims, in, full_dims, full_in, CFL_SIZE);

	// FIXME: more checks
	assert(dims[1] == tdims[1]);
	assert(dims[2] == tdims[2]);

	enum { NC = 3 };

	float qf[NC];	// S in RING

	if (do_ac_adaptive) {

		// Block and Uecker, ISMRM 19:2816 (2001)

		float delays[N];

		if (is_DC && (0 == tdims[1] % 2)) {	
			// Account for asymmetry in DC-sampled trajectory by
			// symmetrization through croppying

			long dims_sym[DIMS];
			md_copy_dims(DIMS, dims_sym, dims);
			dims_sym[PHS1_DIM] -= 1;
			
			long pos[DIMS] = { 0 };
			pos[PHS1_DIM] = 1;

			complex float* in_sym = md_alloc(DIMS, dims_sym, CFL_SIZE);
			md_copy_block(DIMS, pos, dims_sym, in_sym, dims, in, CFL_SIZE);
		
			radial_self_delays(N, delays, angles, dims_sym, in_sym);

		} else
			radial_self_delays(N, delays, angles, dims, in);

		/* We allow an arbitrary quadratic form to account for
		 * non-physical coordinate systems.
		 * Moussavi et al., MRM 71:308-312 (2014)
		 */

		fit_quadratic_form(qf, N, angles, delays);
		
		bart_printf("%f:%f:%f\n", qf[0], qf[1], qf[2]);


	} else if (do_ring) {

		/* RING method
		 * Rosenzweig et al., MRM 81:1898-1906 (2019)
		 */

		ring(&conf, qf, N, angles, dims, in);
		
		bart_printf("%f:%f:%f\n", qf[0], qf[1], qf[2]);

		
	} else {
		
		/* B0 correction method
		 * Moussavi et al., MRM 71:308-312 (2014)
		 */
		assert(argc == 4);

		long b0_dims[2];
		b0_dims[0] = 2;
		b0_dims[1] = full_dims[COIL_DIM];

		complex float* b0 = create_cfl(argv[3], 2, b0_dims);

		// Get DC component of a coil
		long dc_dims[DIMS];
		md_select_dims(DIMS, PHS2_FLAG, dc_dims, full_dims);

		complex float* dc = md_alloc(DIMS, dims, CFL_SIZE);

		long pos1[DIMS] = { 0 };

		assert(full_dims[PHS1_DIM] % 2 == 0);
		pos1[PHS1_DIM] = full_dims[PHS1_DIM] / 2;

		for (unsigned int i = 0; i < full_dims[COIL_DIM]; i++) {

			pos1[COIL_DIM] = i;

			// Phase of DC component
			md_copy_block(DIMS, pos1, dc_dims, dc, full_dims, full_in, CFL_SIZE);
			md_zarg(DIMS, dc_dims, dc, dc);

			float phase[N];

			// Phase unwrap
			phase[0] = crealf(dc[0]);

			for (int j = 1; j < N; j++) {

				phase[j] = crealf(dc[j]);

				if (fabs(phase[0] - phase[j]) > M_PI)
					phase[j] += 2. * M_PI * copysignf(1., phase[0] - phase[j]);
			}

			fit_harmonic(qf, N, angles, phase);

			b0[i * 2 + 0] = qf[0];
			b0[i * 2 + 1] = qf[1];
		}

		md_free(dc);
		unmap_cfl(2, b0_dims, b0);
	}

	if ((do_ac_adaptive || do_ring) && (NULL != argv[3])) {

		long qf_dims[DIMS];
		md_singleton_dims(DIMS, qf_dims);
		qf_dims[0] = NC;

		complex float *pqf = calloc(NC, sizeof(complex float));
		for (int i = 0; i < NC; i++)
			pqf[i] = qf[i] + 0. * I;

		complex float* oqf = create_cfl(argv[3], DIMS, qf_dims);
		md_copy(DIMS, qf_dims, oqf, pqf, sizeof(complex float));

		unmap_cfl(DIMS, qf_dims, oqf);
		xfree(pqf);
	}

	unmap_cfl(DIMS, full_dims, full_in);
	unmap_cfl(DIMS, tdims, traj);

	md_free(in);

	return 0;
}


