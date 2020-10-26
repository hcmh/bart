/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2019 Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2018-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 * 2020 Christian Holme <christian.holme@med.uni-goettingen.de>
 */


/*
 * NOTE: due to the need for compatibility with Siemens IDEA,
 * traj.c and traj.h need to be simultaneously valid C and valid C++!
 */

struct traj_conf {

	_Bool spiral;
	_Bool radial;
	_Bool golden;
	_Bool aligned;
	_Bool full_circle;
	_Bool half_circle_gold;
	_Bool golden_partition;
	_Bool d3d;
	_Bool transverse;
	_Bool asym_traj;
	_Bool mems_traj;
	_Bool rational;
	_Bool sms_turns;
	int accel;
	int tiny_gold;
	int multiple_ga;
};

extern const struct traj_conf traj_defaults;
extern const struct traj_conf rmfreq_defaults;

// for the sequence
enum ePEMode
{
	// PEMODE_CARTESIAN = 1,
	PEMODE_RAD_ALAL = 1,
	PEMODE_RAD_TUAL,
	PEMODE_RAD_GAAL,
	PEMODE_RAD_GA,
	PEMODE_RAD_TUGA,
	PEMODE_RAD_TUTU,
	PEMODE_RAD_RANDAL,
	PEMODE_RAD_RAND,
	PEMODE_RAD_MINV_ALAL,
	PEMODE_RAD_MINV_GA,
	PEMODE_RAD_MEMS_HYB
};

const char* modestr(const enum ePEMode mode);

double seq_rotation_angle(long spoke, long echo, long repetition, long inversion_repetition, long slice, enum ePEMode mode,
		    long num_lines, long num_echoes, long num_repetitions, long num_turns, long num_inv_repets,
		    long num_slices,long tiny_golden_index, long start_pos_GA, bool double_angle);


#ifndef DIMS
#define DIMS 16
#endif

extern void euler(float dir[3], float phi, float psi);
extern void gradient_delay(float d[3], float coeff[2][3], float phi, float psi);
extern void calc_base_angles(double base_angle[DIMS], int Y, int E, int mb, int turns, struct traj_conf conf);
extern bool zpartition_skip(long partitions, long z_usamp[2], long partition, long frame);
extern int gen_fibonacci(int n, int ind);

