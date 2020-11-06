/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2019 Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2018-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
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

#ifndef DIMS
#define DIMS 16
#endif

extern void euler(float dir[3], float phi, float psi);
extern void gradient_delay(float d[3], float coeff[2][3], float phi, float psi);
extern void calc_base_angles(double base_angle[DIMS], int Y, int E, int mb, int turns, struct traj_conf conf);
extern bool zpartition_skip(long partitions, long zusamp[2], long partition, long frame);
extern void z_lookup_fill(int* z_lookup, int z_reflines, int z_npattern, int mb_full, int mb_acc);
extern bool z_contains(int* lookup, int size, int z);
extern int gen_fibonacci(int n, int ind);

