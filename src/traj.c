/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "num/flpmath.h"

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"

static const char usage_str[] = "<output>";
static const char help_str[] = "Computes k-space trajectories.";


static void euler(float dir[3], float phi, float psi)
{
	dir[0] = cosf(phi) * cosf(psi);
	dir[1] = sinf(phi) * cosf(psi);
	dir[2] =             sinf(psi);
}


/* We allow an arbitrary quadratic form to account for
 * non-physical coordinate systems.
 * Moussavi et al., MRM 71:308-312 (2014)
 */
static void gradient_delay(float d[3], float coeff[2][3], float phi, float psi)
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

static void swapElements(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

static uint32_t randomNumber_spcg32(uint64_t s[1]) {
        // Adapted from http://nullprogram.com/blog/2017/09/21/ (Chris Wellons)
        uint64_t m = 0x9b60933458e17d7d;
        uint64_t a = 0xd737232eeccdf7ed;
        *s = *s * m + a;
        int shift = 29 - (*s >> 61);
        return *s >> shift;
}

// Randomly permutation of array elements
static void fisherYates(double arr[], long N)
{
    uint64_t rand_seed[1] = { 476342442 };

    // Swap one-by-one starting from last element
    for (long i = N-1; i > 0; i--)
    {
        // Random number between 0 and i
        uint32_t j = randomNumber_spcg32(rand_seed) % (i+1);

        swapElements(&arr[i], &arr[j]);
    }
}

// z-Undersampling
static bool skip(long dims[DIMS], long zUsamp[3], long partition, long frame)
{
    long zRefLines = zUsamp[0];
    long zAccel = zUsamp[1];

    if (zAccel == 1) { // No undersampling. Do not skip partition.
        return false;
    } else {
	// Auto-Calibration region
        long DC_idx = dims[SLICE_DIM] / 2;
        long AClow_idx = DC_idx - floor(zRefLines/2.);
        long AChigh_idx = DC_idx + ceil(zRefLines/2.) - 1;

        if (partition >= AClow_idx && partition <= AChigh_idx) { // Auto-calibration line. Do not skip partition.
            return false;
        } else { // Check if this non-Auto-calibration line should be sampled.
            long part = (partition < AClow_idx) ? partition : partition - AChigh_idx - 1;

            if (((part - (frame % zAccel)) % zAccel) == 0) {
                   //std::cout << "Don't skip!" << std::endl;
                   return false;
            } else {
                   //std::cout << "Skip partition!" << std::endl;
                   return true;
            }
        }
    }
}

int main_traj(int argc, char* argv[])
{
	int X = 128;
	int Y = 128;
	int mb = 1;
	int accel = 1;
	bool radial = false;
	bool random = false;
	bool golden = false;
	bool aligned = false;
	bool dbl = false;
	bool pGold = false;
	int turns = 1;
	int tinyGold = 0;
	bool d3d = false;
	bool transverse = false;
	bool asymTraj = false;

	float gdelays[2][3] = {
		{ 0., 0., 0. },
		{ 0., 0., 0. }
	};

	long zUsamp[3] = { 0, 1, 0}; // { reference Lines, acceleration, -- }

	const struct opt_s opts[] = {

		OPT_INT('x', &X, "x", "readout samples"),
		OPT_INT('y', &Y, "y", "phase encoding lines"),
		OPT_INT('a', &accel, "a", "acceleration"),
		OPT_INT('t', &turns, "t", "turns"),
		OPT_INT('m', &mb, "mb", "SMS multiband factor"),
		OPT_SET('l', &aligned, "aligned partition angle"),
		OPT_SET('g', &pGold, "golden angle in partition direction"),
		OPT_SET('r', &radial, "radial"),
		OPT_SET('R', &random, "random"),
		OPT_SET('G', &golden, "golden-ratio sampling"),
		OPT_INT('n', &tinyGold, "# Tiny", "tiny golden angle"),
		OPT_SET('D', &dbl, "double base angle"),
		OPT_FLVEC3('q', &gdelays[0], "delays", "gradient delays: x, y, xy"),
		OPT_FLVEC3('Q', &gdelays[1], "delays", "(gradient delays: z, xz, yz)"),
		OPT_SET('O', &transverse, "correct transverse gradient error for radial tajectories"),
		OPT_SET('3', &d3d, "3D"),
		OPT_SET('c', &asymTraj, "Asymmetric trajectory [DC sampled]"),
		OPT_VEC3('z', &zUsamp, "Ref:Acel:-", "Undersampling in z-direction."),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int tot_sp = Y * mb * turns;	// total number of lines/spokes

	int N = X * tot_sp / accel;
	long dims[DIMS] = { [0 ... DIMS - 1] = 1  };
	dims[0] = 3;
	dims[1] = X;
	dims[2] = (radial ? Y : (Y / accel));

	long zRefLines = zUsamp[0];
	long zAccel = zUsamp[1];
	long mb_red;
	if (zAccel > 1) {
		mb_red = zRefLines + (mb - zRefLines) / zAccel;
		if (mb_red < 1)
			error("Invalid z-Acceleration!\n");
	} else
		mb_red = mb;

	if (turns > 1)
		dims[TIME_DIM] = turns;

	if (mb > 1)
		dims[SLICE_DIM] = mb;

	// Checks
	if (d3d){
		if(turns >1)
			error("Turns not implemented for 3D-Couchball\n");
		if(mb > 1)
			error("Multiple partitions not sensible for 3D-Couchball\n");
	}

	if(tinyGold > 1)
		golden = true;

	if (golden) {

		radial = true;

		if(tinyGold == 0)
			tinyGold = 1;

	} else if (dbl || radial) {

		radial = true;

		if ((mb != 1) && (turns != 1))
			if (0 == turns % mb)
				debug_printf(DP_INFO, "Suboptimal spoke distribution: mod(turns,mb) should be nonzero!\n");

	} else {
		if ((turns != 1) || (mb != 1))
			error("Turns or partitinos not allowed/implemented for Cartesian trajectories!");
	}


	complex float* samples = create_cfl(argv[1], DIMS, dims);
	md_clear(DIMS, dims, samples, CFL_SIZE);

	// Angles
       /* Golden-ratio sampling
	* Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O.
	* An optimal radial profile order based on the Golden Ratio
	* for time-resolved MRI. IEEE TMI 26:68--76 (2007)
	*/
	double golden_ratio = (sqrtf(5.) + 1.) / 2;

	/* Tiny golden angle
	* Wundrak S, Paul J, Ulrici J, Hell E, Geibel MA, Bernhardt P, Rottbauer W, Rasche V.
	* Golden ratio sparse MRI using tiny golden angles.
	* Magn Reson Med. 2016 Jun;75(6):2372-8. doi: 10.1002/mrm.25831
	*/
	double golden_angle = M_PI / (golden_ratio + tinyGold - 1);

	// Random angle
	long rand_angle_dims[1];
	double *rand_angle;
	if (random) {
		int no_sp;
		if (aligned) {
			no_sp = (int) tot_sp/mb;
		} else {
			no_sp = tot_sp;
		}
		rand_angle_dims[0] = no_sp;
		rand_angle = md_alloc(1, rand_angle_dims, __SIZEOF_DOUBLE__);
		double angle_inc = 2 * M_PI / no_sp;
		for (long i=0; i<no_sp; i++) {
			rand_angle[i] = i * angle_inc;
		}

		fisherYates(rand_angle, no_sp);
	}

	// Angle between spokes of one slice/partition
	double angle_s;
	angle_s = M_PI / Y * (dbl ? 2 : 1);

	// Angle between turns
	double angle_t;
	if( turns == 1)
		angle_t = 0;
	else
		angle_t = M_PI / (Y * turns * mb_red) * (dbl? 2 : 1);

	// Angle between slices/partitions
	double angle_m;
	if ( mb_red == 1 || aligned)
		angle_m = 0;
	else if (pGold)
		angle_m = 0; // will be explicitly calculated in the loop
	else
		angle_m = M_PI / (Y * mb_red); // linear-turned partitions

	int p = 0;
	// Loop through slices/partitions (m), turns (t), spokes (s) and samples (i)
	for (int m = 0; m < mb; m++){
		for (int t = 0; t < turns; t++){
			if (skip(dims, zUsamp, m, t)) { // Skip this partition for current turn
				p += Y * X;
			} else {
				for(int s = 0; s < Y; s++ ) {
					for (int i = 0; i < X; i++) {
						int j = s + t + m;

						if (radial) {

							/* Calculate read-out samples
							* for symmetric Trajectory [DC between between sample no. X/2-1 and X/2, zero-based indexing]
							* or asymmetric Trajectory [DC component at sample no. X/2, zero-based indexing]
							*/
							double read = (float)i + (asymTraj ? 0 : 0.5) - (float)X / 2.;

							// Angle of current spoke
							double angle = 0;
							if(pGold){
								angle_m = fmod(m * M_PI /  golden_ratio / Y, M_PI / Y);
								angle = s * angle_s + t * angle_t + angle_m;
							}else if (golden){
								if (aligned) {
									angle = golden_angle * ( t * Y + s);
								} else {
									angle = golden_angle * ( t * (Y * mb_red) + s * mb_red + m ); // Increase acquisition angle with each shot chronologically (partitions before turns)
								}
							}else if (random) {
								if (aligned) {
									long current_spoke = t * Y + s;
									angle = rand_angle[current_spoke];
								} else {
									long current_spoke = t * (Y * mb_red) + s * mb_red + m;
									angle = rand_angle[current_spoke];
								}
							} else
								angle = s * angle_s + t * angle_t + m * angle_m;

							// If not double angle, then distribute over half circle
							if (!dbl)
								angle = fmod(angle, M_PI);

							double angle2 = 0.;
							if (d3d) {
								int split = sqrtf(Y);
								angle2 = s * angle_s * split;
							}


							float d[3] = { 0., 0., 0 };
							gradient_delay(d, gdelays, angle, angle2);

							float read_dir[3];
							euler(read_dir, angle, angle2);

							if (!transverse) {

								// project to read direction

								float delay = 0.;

								for (unsigned int i = 0; i < 3; i++)
									delay += read_dir[i] * d[i];

								for (unsigned int i = 0; i < 3; i++)
									d[i] = delay * read_dir[i];
							}

							samples[p * 3 + 0] = d[1] + read * read_dir[1];
							samples[p * 3 + 1] = d[0] + read * read_dir[0];
							samples[p * 3 + 2] = d[2] + read * read_dir[2];

						} else {

							samples[p * 3 + 0] = (i - X / 2);
							samples[p * 3 + 1] = (j - Y / 2);
							samples[p * 3 + 2] = 0;
						}

						p++;
					}
				}
			}
		}
	}
	assert(p == N - 0);

	unmap_cfl(3, dims, samples);
	if (random)
		md_free(rand_angle);
	exit(0);
}


