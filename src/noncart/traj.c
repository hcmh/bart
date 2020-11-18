/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2020 Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2014-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <complex.h>
#include "num/rand.h"


#ifdef SSAFARY_PAPER
#include "misc/debug.h"
#endif

#include "misc/mri.h"
#ifdef SSAFARY_PAPER
#include "misc/debug.h"
#endif

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
	.accel = 1,
	.tiny_gold = 0,
	.rational = false,
	.multiple_ga = 1,
	.sms_turns = false,
};

const struct traj_conf rmfreq_defaults = {

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
	.accel = 1,
	.tiny_gold = 0,
	.rational = false,
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

	/* radial multi-echo multi-spoke sampling
	 *
	 * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
	 * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
	 * estimation using undersampled  triple-echo multi-spoke radial FLASH.
	 * Magn Reson Med 82:1000-1011 (2019)
	 */
	double angle_e = 0.;

	if (conf.mems_traj) {

		angle_s = angle_s * 1.;
		angle_e = angle_s / E + M_PI;
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
bool zpartition_skip(long partitions, long zusamp[2], long partition, long frame)
{
	long z_reflines = zusamp[0];
	long z_acc = zusamp[1];

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


// check if slice/partition z is part of AC region
static bool z_in_AC(int z_reflines, int mb, int z)
{
    long DC_idx = mb / 2;
    long AC_low_idx = DC_idx - (long)floor(z_reflines/2.);
    long AC_high_idx = DC_idx + (long)ceil(z_reflines/2.) - 1;

    if (z >= AC_low_idx && z <= AC_high_idx) // z is part of AC region
        return true;
    else
        return false;
}

// check if lookup table contains slice/partition z
bool z_contains(int* lookup, int size, int z)
{
    for ( int i = 0; i < size; i++ )
        if ( lookup[i] == z )
            return true;

    return false;
}

// fill z-undersampling lookup table
void z_lookup_fill(int* z_lookup, int z_reflines, int z_npattern, int mb_full, int mb_acc)
{
	for (int i = 0; i < (z_reflines * z_npattern); i++)
		z_lookup[i] = -1;

	// insert reference lines
    for (int i = 0; i < z_npattern; i++) {

        int count = 0;
        for (int j = 0; j < mb_full; j++) {

            if (z_in_AC(z_reflines, mb_full, j)) { // j is a reference line

                    z_lookup[(i * mb_acc) + count] = j;
                    count++;
            }
        }
    }

	// fill remaining entries with random values
    uint64_t rand_seed[1] = { 476342442 };

    double mean = mb_full / 2.;
    double stdv = mb_full / 4.;

    for ( int i = 0; i < z_npattern; i++ ) {

        int count = 0; // prevent infinite loop
        for ( int z = z_reflines; z < mb_acc; z++ ) {

            do {
				int z_new = -1;

				do {

					z_new = (int)(creal(rand_spcg32_normal(rand_seed)) * stdv + mean);
				} while (z_new < 0 || z_new >= mb_full);

				z_lookup[(i * mb_acc) + z] = z_new;
                count ++;

            } while (z_contains(&z_lookup[i * mb_acc], z, z_lookup[(i * mb_acc) + z]) && count < 100);

            if (count == 100)
                z_lookup[(i * mb_acc) + z] = (int)mean;
        }
    }


}

