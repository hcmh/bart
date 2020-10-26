/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2020 Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2014-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 * 2020 Christian Holme <christian.holme@med.uni-goettingen.de>
 */


/*
 * NOTE: due to the need for compatibility with Siemens IDEA,
 * traj.c and traj.h need to be simultaneously valid C and valid C++!
 */

#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "traj.h"

const struct traj_conf traj_defaults = {

	.spiral = false,
	.radial = false,
	.golden = false,
	.aligned = false,
	.full_circle = false,
	.half_circle_gold = false,
	.golden_partition = false,
	.d3d = false,
	.transverse = false,
	.asym_traj = false,
	.mems_traj = false,
	.rational = false,
	.sms_turns = false,
	.accel = 1,
	.tiny_gold = 0,
	.multiple_ga = 1,
};

const struct traj_conf rmfreq_defaults = {

	.spiral = false,
	.radial = true,
	.golden = false,
	.aligned = false,
	.full_circle = false,
	.half_circle_gold = false,
	.golden_partition = false,
	.d3d = false,
	.transverse = false,
	.asym_traj = false,
	.mems_traj = false,
	.rational = false,
	.sms_turns = false,
	.accel = 1,
	.tiny_gold = 0,
	.multiple_ga = 1,
};


void euler(float dir[3], float phi, float psi)
{
	dir[0] = cosf(phi) * cosf(psi);
	dir[1] = sinf(phi) * cosf(psi);
	dir[2] =             sinf(psi);
}


/* We allow an arbitrary quadratic form to account for
 * non-physical coordinate systems.
 * Moussavi et al., MRM 71:308-312 (2014)
 */
void gradient_delay(float d[3], float coeff[2][3], float phi, float psi)
{
	float dir[3];
	euler(dir, phi, psi);

	float mat[3][3] = {

		{ coeff[0][0], coeff[0][2], coeff[1][1] },
		{ coeff[0][2], coeff[0][1], coeff[1][2] },
		{ coeff[1][1], coeff[1][2], coeff[1][0] },
	};

	for (unsigned int i = 0; i < 3; i++) {

		d[i] = 0.;

		for (unsigned int j = 0; j < 3; j++)
			d[i] += mat[i][j] * dir[j];
	}
}

static void fib_next(int f[2])
{
	int t = f[0];
	f[0] = f[1];
	f[1] += t;
}

int gen_fibonacci(int n, int ind)
{
	int fib[2] = { 1, n };

	for (int i = 0; i < ind; i++)
		fib_next(fib);

	return fib[1];
}

static double rational_angle(int Y, int n)
{
	int fib1[2] = { 1, 1 };
	int fibn[2] = { 1, n };

	while (fibn[1] < Y) {

		fib_next(fib1);
		fib_next(fibn);
	}

	return M_PI  * fib1[0] / fibn[1];
}


void calc_base_angles(double base_angle[DIMS], int Y, int E, int mb, int turns, struct traj_conf conf)
{
	/*
	 * Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O.
	 * An optimal radial profile order based on the Golden Ratio
	 * for time-resolved MRI. IEEE TMI 26:68--76 (2007)
	 *
	 * Wundrak S, Paul J, Ulrici J, Hell E, Geibel MA, Bernhardt P, Rottbauer W, Rasche V.
	 * Golden ratio sparse MRI using tiny golden angles.
	 * Magn Reson Med 75:2372-2378 (2016)
	 */

	double golden_ratio = (sqrt(5.) + 1.) / 2;
	double golden_angle = M_PI / (golden_ratio + conf.tiny_gold - 1.);

	// For numerical stability
	if (1 == conf.tiny_gold)
		golden_angle = M_PI * (2. - (3. - sqrt(5.))) / 2.;

	double angle_atom = M_PI / Y;

	if (conf.rational)
		golden_angle = rational_angle(Y, conf.tiny_gold);

	golden_angle *= conf.multiple_ga;

	// Angle between spokes of one slice/partition
	double angle_s = angle_atom * (conf.full_circle ? 2 : 1);

	// Angle between slices/partitions
	double angle_m = angle_atom / mb; // linear-turned partitions

	if (conf.aligned)
		angle_m = 0;

	// Angle between turns
	double angle_t = 0.;

	if (turns > 1)
		angle_t = angle_atom / (turns * (conf.sms_turns ? mb : 1)) * (conf.full_circle ? 2 : 1);


	double angle_e = M_PI; // flip all even echoes in any kind of multi-echo

	/* radial multi-echo multi-spoke sampling
	 *
	 * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
	 * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
	 * estimation using undersampled  triple-echo multi-spoke radial FLASH.
	 * Magn Reson Med 82:1000-1011 (2019)
	 */
	if (conf.mems_traj) {

		angle_s = angle_s * 1.;
		angle_e += angle_s / E;
		angle_t = golden_angle;

	} else if (conf.golden) {

		angle_s = golden_angle;
		angle_m = 0;
		angle_t = golden_angle * Y;

		// Continuous golden angle with multiple partitions/slices
		if ((mb > 1) && (!conf.aligned)) {

			angle_m = golden_angle;
			angle_s = golden_angle * mb;
			angle_t = golden_angle * mb * Y;

		}

#ifdef SSAFARY_PAPER
		/* Specific trajectory designed for z-undersampled Stack-of-Stars imaging:
		 *
		 * Sebastian Rosenzweig, Nick Scholand, H. Christian M. Holme, Martin Uecker.
		 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular Spectrum
		 * Analysis (SSA-FARY). IEEE Tran Med Imag 2020; 10.1109/TMI.2020.2985994. arXiv:1812.09057.
		 */

		if (14 == mb) {

			int mb_red = 8;

			angle_m = golden_angle;
			angle_s = golden_angle * mb_red;
			angle_t = golden_angle * Y * mb_red;

			debug_printf(DP_INFO, "Trajectory generation to reproduce SSA-FARY Paper!\n");
		}
#endif
	}

	base_angle[PHS2_DIM] = angle_s;
	base_angle[SLICE_DIM] = angle_m;
	base_angle[TE_DIM] = angle_e;
	base_angle[TIME_DIM] = angle_t;
}


// z-Undersampling
bool zpartition_skip(long partitions, long z_usamp[2], long partition, long frame)
{
	long z_reflines = z_usamp[0];
	long z_acc = z_usamp[1];

	if (1 == z_acc) // No undersampling. Do not skip partition
		return false;


	// Auto-Calibration region

	long DC_idx = partitions / 2;
	long AC_lowidx = DC_idx - floor(z_reflines / 2.);
	long AC_highidx = DC_idx + ceil(z_reflines / 2.) - 1;

	if ((partition >= AC_lowidx) && (partition <= AC_highidx)) // Auto-calibration line. Do not skip partition.
		return false;

	// Check if this non-Auto-calibration line should be sampled.

	long part = (partition < AC_lowidx) ? partition : (partition - AC_highidx - 1);

	if (0 == ((part - (frame % z_acc)) % z_acc))
		return false;

	return true;
}


const char* modestr(const enum ePEMode mode)
{
	switch(mode)
	{
		case PEMODE_RAD_ALAL:
			return "ALAL";
		case PEMODE_RAD_TUAL:
			return "TUAL";
		case PEMODE_RAD_GAAL:
			return "GAAL";
		case PEMODE_RAD_GA:
			return "GA";
		case PEMODE_RAD_TUGA:
			return "TUGA";
		case PEMODE_RAD_TUTU:
			return "TUTU";
		case PEMODE_RAD_RANDAL:
		case PEMODE_RAD_RAND:
			return "INVALID! RAND!";
		case PEMODE_RAD_MINV_ALAL:
			return "MINV_ALAL";
		case PEMODE_RAD_MINV_GA:
			return "MINV_GA";
		case PEMODE_RAD_MEMS_HYB:
			return "MEMS";
		default:
			return "INVALID!";
	}
}


double seq_rotation_angle(long spoke, long echo, long repetition, long inversion_repetition, long slice, enum ePEMode mode, long num_lines, long num_echoes, long num_repetitions,
		    long num_turns, long num_inv_repets, long num_slices, long tiny_golden_index, long start_pos_GA, bool double_angle)
{

	double angle_spoke = 0.; // increment angle for spoke [rad]
	double angle_frame = 0.; // increment angle for frame [rad]
	double angle_slice = 0.; // increment angle for slice [rad]
	double angle_echo  = 0.; // increment angle for echo [rad]

	double base_angle[16] = {0.f};

	struct traj_conf conf = {.spiral = false, .radial = true, .golden = false, .aligned = true, .full_circle = double_angle,
		.half_circle_gold = false, .golden_partition = false, .d3d = false, .transverse = false, .asym_traj = false,
		.mems_traj = false, .rational = false, .sms_turns = true, .accel = 1, .tiny_gold = (int) tiny_golden_index, .multiple_ga = 1};

	switch (mode)
	{

		// Radial | Aligned frames | Aligned partitions
		case PEMODE_RAD_ALAL:
		// Radial | Turned frames | Aligned partitions
		case PEMODE_RAD_TUAL:
			conf.sms_turns = false;
			break;
		// Radial | Turned frames | (Linear)-turned partitions
		case PEMODE_RAD_TUTU:
			conf.aligned = false;
			break;

		// Radial | Turned frames | Golden-angle partitions
		case PEMODE_RAD_TUGA:
			conf.aligned = false;
			conf.golden_partition = true;
			conf.sms_turns = false;
			break;

		// Radial | Golden-angle frames | Aligned partitions
		case PEMODE_RAD_GAAL:
			conf.golden = true;
			break;

		// Radial | Consecutive spokes aquired in GA fashion
		case PEMODE_RAD_GA:
			conf.golden = true;
			conf.golden_partition = true;
			conf.aligned = false;
			break;

		// Radial | Multiple inversion recovery |
		case PEMODE_RAD_MINV_ALAL:
			num_lines = num_inv_repets; 	// TODO: Very ugly, but necessary
							// to calculate the correct angle
			break;

		// Radial | Multiple inversion recovery |
		case PEMODE_RAD_MINV_GA:
			conf.golden = true;
			conf.golden_partition = true;
			conf.aligned = false;
			break;

		// Radial | Multi-Echo Multi-Spoke Hybrid
		//   1) uniform spoke distribution within one frame
		//   2) golden angle increment between frames and partitions
		// Refer to:
		// * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
		// * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
		// * estimation using undersampled triple-echo multi-spoke radial FLASH.
		// * Magn Reson Med 82:1000-1011 (2019)
		case PEMODE_RAD_MEMS_HYB:
			conf.mems_traj = true;
			break;

		case PEMODE_RAD_RAND:
		case PEMODE_RAD_RANDAL: // No longer implemented! // Radial | Randomly aligned
		default:
			return 0.f;
			break;
	}

	calc_base_angles(base_angle, num_lines, num_echoes, num_slices, num_turns, conf);

	angle_spoke = base_angle[2];
	angle_echo = base_angle[5];
	angle_frame = base_angle[10];
	angle_slice = base_angle[13];

	long tspoke = spoke;
	long tframe = repetition;
	long tslice = slice;
	long techo = echo;


	switch (mode)
	{
		// Radial | Aligned frames | Aligned partitions
		case PEMODE_RAD_ALAL:
		// Radial | Turned frames | Aligned partitions
		case PEMODE_RAD_TUAL:
		// Radial | Turned frames | (Linear)-turned partitions
		case PEMODE_RAD_TUTU:
			tframe = (repetition % num_turns);
			break;

		// Radial | Turned frames | Golden-angle partitions // DEPRECATED
		case PEMODE_RAD_TUGA: {
			tframe = (repetition % num_turns);
			double golden_ratio = (sqrt(5.) + 1.) / 2;
			if (0 < slice)
				angle_slice = fmod( (double)slice * (M_PI / (double)num_lines) / golden_ratio, M_PI / (double)num_lines ) / slice;
			else
				angle_slice = 0.f;
			break;
		}

		// Radial | Golden-angle frames | Aligned partitions
		case PEMODE_RAD_GAAL:
			tframe = (start_pos_GA + repetition);
			break;

		// Radial | Consecutive spokes aquired in GA fashion
		case PEMODE_RAD_GA:
			break;

		// Radial | Multiple inversion recovery |
		case PEMODE_RAD_MINV_ALAL:
			tspoke = inversion_repetition;
			tframe = 0;
			tslice = 0;
			techo = 0;
			break;

		// Radial | Multiple inversion recovery |
		case PEMODE_RAD_MINV_GA:
			tframe = ( repetition + inversion_repetition * (num_repetitions+1) );
			break;

		// Radial | Multi-Echo Multi-Spoke Hybrid
		case PEMODE_RAD_MEMS_HYB:
			break;

		case PEMODE_RAD_RAND:
		case PEMODE_RAD_RANDAL: // No longer implemented! // Radial | Randomly aligned
		default:
			break;
	}

	double phi = angle_spoke * tspoke
		+ angle_frame * tframe
		+ angle_slice * tslice
		+ angle_echo * techo;

	return phi;
}
