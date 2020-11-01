/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2019 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019 Aurélien Trotier <a.trotier@gmail.com>
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
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

#include "noncart/traj.h"

static const char usage_str[] = "<output>";
static const char help_str[] = "Computes k-space trajectories.";


int main_traj(int argc, char* argv[argc])
{
	int X = 128;
	int Y = 128;
	int D = -1;
	int E = 1;
	int mb = 1;
	int turns = 1;
	int even_echo_shift = 0;
	float rot = 0.;
	bool force_grid = false;


	struct traj_conf conf = traj_defaults;

	float gdelays[2][3] = {
		{ 0., 0., 0. },
		{ 0., 0., 0. }
	};

	long zusamp[2] = { 0, 1 }; // { reference Lines, acceleration }

	const char* custom_angle = NULL;
	const char* custom_gdelays = NULL;


	const struct opt_s opts[] = {

		OPT_INT('x', &X, "x", "actual readout samples"),
		OPT_INT('y', &Y, "y", "phase encoding lines"),
		OPT_INT('d', &D, "d", "full readout samples"),
		OPT_INT('e', &E, "e", "number of echoes"),
		OPT_INT('a', &conf.accel, "a", "acceleration"),
		OPT_INT('t', &turns, "t", "turns"),
		OPT_INT('m', &mb, "mb", "SMS multiband factor"),
		OPT_INT('v', &even_echo_shift, "(shift)", "(even echo shift)"),
		OPT_SET('l', &conf.aligned, "aligned partition angle"),
		OPT_SET('g', &conf.golden_partition, "(golden angle in partition direction)"),
		OPT_SET('r', &conf.radial, "radial"),
		OPT_SET('G', &conf.golden, "golden-ratio sampling"),
		OPT_SET('H', &conf.half_circle_gold, "(halfCircle golden-ratio sampling)"),
		OPT_INT('s', &conf.tiny_gold, "# Tiny GA", "tiny golden angle"),
		OPT_INT('M', &conf.multiple_ga, "# Multiple", "multiple golden angle"),
		OPT_SET('A', &conf.rational, "rational approximation of golden angles"),
		OPT_SET('D', &conf.full_circle, "projection angle in [0,360°), else in [0,180°)"),
		OPT_FLOAT('R', &rot, "phi", "rotate"),
		OPT_FLVEC3('q', &gdelays[0], "delays", "gradient delays: x, y, xy"),
		OPT_FLVEC3('Q', &gdelays[1], "delays", "(gradient delays: z, xz, yz)"),
		OPT_SET('O', &conf.transverse, "correct transverse gradient error for radial tajectories"),
		OPT_SET('3', &conf.d3d, "3D"),
		OPT_SET('c', &conf.asym_traj, "asymmetric trajectory"),
		OPT_SET('E', &conf.mems_traj, "multi-echo multi-spoke trajectory"),
		OPT_VEC2('z', &zusamp, "Ref:Acel", "Undersampling in z-direction."),
		OPT_STRING('C', &custom_angle, "file", "custom_angle file [phi + i * psi]"),
		OPT_STRING('V', &custom_gdelays, "file", "custom_gdelays"),
		OPT_SET('T', &conf.sms_turns, "(Modified SMS Turn Scheme)"),
		OPT_SET('f', &force_grid, "Force trajectory samples on Cartesian grid."),
		OPT_SET('S', &conf.spiral, "Archimedean spiral readout"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	// Load custom_angle
	long sdims[DIMS];
	complex float* custom_angle_val = NULL;

	if ((NULL != custom_angle) && conf.radial) {

		debug_printf(DP_INFO, "custom_angle file is used \n");

		custom_angle_val = load_cfl(custom_angle, DIMS, sdims);

		if (Y != sdims[0]) {

			debug_printf(DP_INFO, "According to the custom angle file : number of projection (y) = %d\n", sdims[0]);
			Y = sdims[0];
		}
	}

	if (conf.rational) {

		conf.golden = true;

		int i = 0;
		int spokes = M_PI / 2. * X;

		while (spokes > gen_fibonacci(conf.tiny_gold, i))
			i++;

		int total = gen_fibonacci(conf.tiny_gold, i);

		debug_printf(DP_INFO, "Rational approximation golden angle sampling:\n");
		debug_printf(DP_INFO, "Optimal number of spokes: %d (Nyquist: %d).\n", total, spokes);
		debug_printf(DP_INFO, "Base angle (full circle): %f = 2 pi / %d\n", 2. * M_PI / total, total);
		debug_printf(DP_INFO, "Index increment per spoke: %d\n", gen_fibonacci(0, i - 1));
		debug_printf(DP_INFO, "Index for spoke n: (n * %d) mod %d\n", gen_fibonacci(0, i - 1), total);
		debug_printf(DP_INFO, "Angle for spoke n: ((n * %d) mod %d) * %f\n", gen_fibonacci(0, i - 1), total, 2. * M_PI / total);
	}

	// variables for z-undersampling
	long z_reflines = zusamp[0];
	long z_acc = zusamp[1];
	int mb2 = mb / z_acc;

    if ( z_reflines > mb2)
		error("ERROR: More reference lines than partitions/slices!\n");



	int tot_sp = Y * E * mb2 * turns;	// total number of lines/spokes
	int N = X * tot_sp / conf.accel;


	long dims[DIMS] = { [0 ... DIMS - 1] = 1  };
	dims[0] = 3;
	dims[1] = X;
	dims[2] = (conf.radial ? Y : (Y / conf.accel));

	dims[TE_DIM] = E;

	if (conf.mems_traj)
		conf.radial = true;

	if (-1 == D)
		D = X;
	
	if (D < X)
	    error("actual readout samples must be less than full samples");


	dims[TIME_DIM] = turns;
	dims[SLICE_DIM] = mb2;

	if (conf.half_circle_gold) {

		debug_printf(DP_WARN, "half-circle golden angle (-H) is deprecated and might be removed in a future version!\n");

		conf.golden = true;

		if (conf.full_circle)
			error("Invalid options. Full-circle or half-circle sampling?");
	}


	if (conf.d3d) {

		if (turns >1)
			error("Turns not implemented for 3D-Kooshball\n");

		if (mb > 1)
			error("Multiple partitions not sensible for 3D-Kooshball\n");
	}

	if (conf.radial && conf.spiral)
		error("Choose either radial spokes or spirals\n");

	if (conf.golden_partition)
		debug_printf(DP_WARN, "golden partitions (-g) is deprecated and might be removed in a future version!\n");

	if (conf.tiny_gold >= 1)
		conf.golden = true;

	if (conf.golden) {

		if (!conf.spiral)
			conf.radial = true;

		if (0 == conf.tiny_gold)
			conf.tiny_gold = 1;

	} else if ( (conf.full_circle && !conf.spiral) || conf.radial) {

		conf.radial = true;

	} else { // Cartesian

		if ((turns != 1) || (mb != 1))
			error("Turns or partitions not allowed/implemented for Cartesian trajectories!");
	}



	long gdims[DIMS] = { 0 };
	complex float* custom_gdelays_val = (NULL!=custom_gdelays) ? load_cfl(custom_gdelays, DIMS, gdims) : NULL;



	complex float* samples = (z_acc > 1) ? anon_cfl(NULL, DIMS, dims) : create_cfl(argv[1], DIMS, dims);

	md_clear(DIMS, dims, samples, CFL_SIZE);

	double base_angle[DIMS] = { 0. };
	calc_base_angles(base_angle, Y, E, mb2, turns, conf);

	int p = 0;
	long pos[DIMS] = { 0 };

	do {
		int i = pos[PHS1_DIM];
		int j = pos[PHS2_DIM] * conf.accel;
		int e = pos[TE_DIM];
		int m = pos[SLICE_DIM];

		if (conf.radial) {

			int s = j;

			/* Calculate read-out samples
			 * for symmetric trajectory [DC between between sample no. X/2-1 and X/2, zero-based indexing]
			 * or asymmetric trajectory [DC component at sample no. X/2, zero-based indexing]
			 */
			double read = (float)(i + D - X) + (conf.asym_traj ? 0 : 0.5) - (float)D / 2.;

			// For odd echos in asymmetric multi-echo, the DC component is later than half of the readout instead of earlier.
			// Therefore, for such echos, the readout position needs to be calculated as follows:
			if ((D != X) && (1 == (e % 2)))
					read = (float)i + (conf.asym_traj ? 0. : 0.5) - (float)D / 2.;

			if (conf.golden_partition) {

				double golden_ratio = (sqrt(5.) + 1.) / 2;
				double angle_atom = M_PI / Y;

				base_angle[SLICE_DIM] = (m > 0) ? (fmod(angle_atom * m / golden_ratio, angle_atom) / m) : 0;
			}

			double angle = 0.;

			for (unsigned int d = 1; d < DIMS; d++)
				angle += pos[d] * base_angle[d];


			if (conf.half_circle_gold)
				angle = fmod(angle, M_PI);

			angle += M_PI * rot / 180.;

			// 3D
			double angle2 = 0.;

			if (conf.d3d) {

				int split = sqrt(Y);
				angle2 = s * M_PI / Y * (conf.full_circle ? 2 : 1) * split;

				if (NULL != custom_angle)
					angle2 = cimag(custom_angle_val[j]);
			}


			if (NULL != custom_angle)
				angle = creal(custom_angle_val[j]);


			if (NULL != custom_gdelays) {

				long eind = e * 3;

				for (int qind = 0; qind < 3; qind++) {
					gdelays[0][qind] = creal(custom_gdelays_val[eind+qind]);
				}
			}

			float d[3] = { 0., 0., 0 };
			gradient_delay(d, gdelays, angle, angle2);

			float read_dir[3];
			euler(read_dir, angle, angle2);

			if (!conf.transverse) {

				// project to read direction

				float delay = 0.;

				for (unsigned int i = 0; i < 3; i++)
					delay += read_dir[i] * d[i];

				for (unsigned int i = 0; i < 3; i++)
					d[i] = delay * read_dir[i];
			}

			samples[p * 3 + 0] = (force_grid) ? (int) (d[1] + read * read_dir[1]) : d[1] + read * read_dir[1];
			samples[p * 3 + 1] = (force_grid) ? (int) (d[0] + read * read_dir[0]) : d[0] + read * read_dir[0];
			samples[p * 3 + 2] = (force_grid) ? (int) (d[2] + read * read_dir[2]) : d[2] + read * read_dir[2];

		} else if (conf.spiral) {

			// supports Archimedean spirals

			double read = (float)(i + D - X);

			double steps = D;	// number of steps along whole spiral
			double offset = 0.;
			double density = 2.;
			double cycles = X / (4 * M_PI * density); // whole k-space coverage: X/2 = density * cycles * 2 Pi

			double incr = read / steps * cycles * 2 * M_PI;

			double angle = 0.;

			for (unsigned int d = 1; d < DIMS; d++)
				angle += pos[d] * base_angle[d];

			samples[p * 3 + 0] = (offset + density * incr) * cosf(incr + angle);
			samples[p * 3 + 1] = (offset + density * incr) * sinf(incr + angle);
			samples[p * 3 + 2] = 0;

		} else {

			samples[p * 3 + 0] = (i - X / 2);
			samples[p * 3 + 1] = (j - Y / 2);
			samples[p * 3 + 2] = 0;
		}

		p++;

	} while(md_next(DIMS, dims, ~1L, pos));

	assert(p == N - 0);

	if (NULL != custom_gdelays)
		unmap_cfl(DIMS, gdims, custom_gdelays_val);


	if (NULL != custom_angle_val)
		unmap_cfl(3, sdims, custom_angle_val);
	
	// "inflate" partition dimension when z-undersampling is used
	if (z_acc > 1) {

		long pos[DIMS] = { 0 };

		// actual trajectory
		long z_dims[DIMS];
		for (unsigned int i = 0; i < DIMS; i++)
			z_dims[i] = dims[i];
		z_dims[SLICE_DIM] = mb;

		long z_strs[DIMS];
		md_calc_strides(DIMS, z_strs, z_dims, CFL_SIZE); 

		complex float* traj = create_cfl(argv[1], DIMS, z_dims);
		md_clear(DIMS, z_dims, traj, CFL_SIZE);

		// initialize lookup table which contains the indices of the partitions/slices to be sampled
		const int z_npattern_max = 500;
		const int z_npattern = ((turns * Y) > z_npattern_max) ? z_npattern_max : (turns * Y);

		int* z_lookup = (int*)malloc((z_npattern * mb2) * sizeof(int));
		z_lookup_fill(z_lookup, z_reflines, z_npattern, mb, mb2);

		// copy slices/partitions of 'samples'-array to correct indices of 'traj'-array
		long dims_red[DIMS];
		md_select_dims(DIMS, ~(PHS2_FLAG|SLICE_FLAG|TIME_FLAG), dims_red, dims);

		long z_pos_src[DIMS] = { 0 };
		long z_pos_dst[DIMS] = { 0 };
		int offset = 0;
		int s, t;

		do {
			s = pos[PHS2_DIM];
			t = pos[TIME_DIM];
			z_pos_src[TIME_DIM] = t;
			z_pos_dst[TIME_DIM] = t;
			z_pos_src[PHS2_DIM] = s;
			z_pos_dst[PHS2_DIM] = s;

			for (int z = 0; z < z_dims[SLICE_DIM]; z++) {

				z_pos_dst[SLICE_DIM] = z;
				offset = ((s + t * dims[PHS2_DIM]) * mb2) % (z_npattern * mb2);
				if (z_contains(&z_lookup[offset], mb2, z)) {
					md_copy_block(DIMS, z_pos_src, dims_red, &MD_ACCESS(DIMS, z_strs, z_pos_dst, traj), dims, samples, CFL_SIZE);			
					assert(z_pos_src[SLICE_DIM]++ < dims[SLICE_DIM]);
				}
			}

			z_pos_src[SLICE_DIM] = 0;
			
		} while(md_next(DIMS, dims, PHS2_FLAG|TIME_FLAG, pos));

		unmap_cfl(3, z_dims, traj);
		free(z_lookup);
	}

	unmap_cfl(3, dims, samples);

	exit(0);
}



