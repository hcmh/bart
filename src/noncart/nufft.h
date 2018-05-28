/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

struct operator_s;
struct linop_s;

struct nufft_conf_s {

	_Bool toeplitz; ///< Toeplitz embedding boolean for A^T A
	_Bool pcycle; /// < Phase cycling
	_Bool periodic;
};

extern struct nufft_conf_s nufft_conf_defaults;


extern struct linop_s* nufft_create(unsigned int N,			///< Number of dimensions
				    const long ksp_dims[__VLA(N)],	///< Kspace dimension
				    const long coilim_dims[__VLA(N)],	///< Coil image dimension
				    const long traj_dims[__VLA(N)],	///< Trajectory dimension
				    const _Complex float* traj,		///< Trajectory
				    const _Complex float* weights,	///< Weights, ex, density-compensation
				    struct nufft_conf_s conf);		///< NUFFT configuration


extern _Complex float* compute_psf(unsigned int N,
				   const long img2_dims[__VLA(N)],
				   const long trj_dims[__VLA(N)],
				   const complex float* traj,
				   const complex float* weights);

extern void estimate_im_dims(int N, unsigned long flags, long dims[__VLA(N)], const long tdims[__VLA(N)], const complex float* traj);

#include "misc/cppwrap.h"

