/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Martin Heide
 */

#ifndef GRID_H
#define GRID_H 1

#include <complex.h>
#include <stdbool.h>

#define VEC_DIM_S READ_DIM
#define VEC_FLAG_S (1u << VEC_DIM_S)

struct grid_opts {

	long dims[DIMS];
	float veclen[4];
	bool kspace;
	float b0[3];
	float b1[3];
	float b2[3];
	float bt;
};

extern struct grid_opts grid_opts_defaults;

extern float* compute_grid(int D, long gdims[D], struct grid_opts* go, const long tdims[D], const _Complex float* traj);

#endif

