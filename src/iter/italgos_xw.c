/* Copyright 2013-2017. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013-2014 Frank Ong <frankong@berkeley.edu>
 * 2013-2014,2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 *
 * Landweber L. An iteration formula for Fredholm integral equations of the
 * first kind. Amer. J. Math. 1951; 73, 615-624.
 *
 * Nesterov Y. A method of solving a convex programming problem with
 * convergence rate O (1/k2). Soviet Mathematics Doklady 1983; 27(2):372-376
 *
 * Bakushinsky AB. Iterative methods for nonlinear operator equations without 
 * regularity. New approach. In Dokl. Russian Acad. Sci 1993; 330:282-284.
 *
 * Daubechies I, Defrise M, De Mol C. An iterative thresholding algorithm for
 * linear inverse problems with a sparsity constraint. 
 * Comm Pure Appl Math 2004; 57:1413-1457.
 *
 * Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm for
 * linear inverse problems. SIAM Journal on Imaging Sciences 2.1 2009; 183-202.
 */

#include <math.h>
#include <stdbool.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "iter/vec.h"
#include "iter/monitor.h"
#include "iter/italgos.h"

#include "italgos_xw.h"

#define MPI_CGtol
// #define autoScaling

/**
 * ravine step
 * (Nesterov 1983)
 */
static void ravine(const struct vec_iter_s* vops, long N, float* ftp, float* xa, float* xb)
{
	float ft = *ftp;
	float tfo = ft;

	ft = (1.f + sqrtf(1.f + 4.f * ft * ft)) / 2.f;
	*ftp = ft;

	vops->swap(N, xa, xb);
	vops->axpy(N, xa, (1.f - tfo) / ft - 1.f, xa);
	vops->axpy(N, xa, (tfo - 1.f) / ft + 1.f, xb);
}



/**
 * Store information about iterative algorithm.
 * Used to flexibly modify behavior, e.g. continuation
 *
 * @param rsnew current residual
 * @param rsnot initial residual
 * @param iter current iteration
 * @param maxiter maximum iteration
 */
struct iter_data {

	double rsnew;
	double rsnot;
	unsigned int iter;
	const unsigned int maxiter;
};



/**
 * Continuation for regularization. Returns fraction to scale regularization parameter
 *
 * @param itrdata state of iterative algorithm
 * @param delta scaling of regularization in the final iteration (1. means don't scale, 0. means scale to zero)
 *
 */
static float ist_continuation(struct iter_data* itrdata, const float delta)
{
/*
	// for now, just divide into evenly spaced bins
	const float num_steps = itrdata->maxiter - 1;

	int step = (int)(itrdata->iter * num_steps / (itrdata->maxiter - 1));

	float scale = 1. - (1. - delta) * step / num_steps;

	return scale;
*/
	float a = logf( delta ) / (float) itrdata->maxiter;
	return expf( a * itrdata->iter );
}




/**
 * Iterative Soft Thresholding/FISTA to solve min || b - Ax ||_2 + lambda || T x ||_1
 *
 * @param maxiter maximum number of iterations
 * @param epsilon stop criterion
 * @param tau (step size) weighting on the residual term, A^H (b - Ax)
 * @param lambda_start initial regularization weighting
 * @param lambda_end final regularization weighting (for continuation)
 * @param N size of input, x
 * @param vops vector ops definition
 * @param op linear operator, e.g. A
 * @param thresh threshold function, e.g. complex soft threshold
 * @param x initial estimate
 * @param b observations
 */
void fista_xw(unsigned int maxiter, float epsilon, float tau, long* dims,
	float continuation, bool hogwild,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor)
{

	struct iter_data itrdata = {

		.rsnew = 1.,
		.rsnot = 1.,
		.iter = 0,
		.maxiter = maxiter,
	};

	float* r = vops->allocate(N);
	float* o = vops->allocate(N);

	float ra = 1.;
	vops->copy(N, o, x);

	itrdata.rsnot = vops->norm(N, b);

	float ls_old = 1.;
	float lambda_scale = 1.;

	int hogwild_k = 0;
	int hogwild_K = 10;

	long res = dims[0];
	long parameters = dims[COEFF_DIM];
	long SMS = dims[SLICE_DIM];
	long TIME2 = dims[TIME2_DIM];
	int temp_index;
	unsigned int u,v,w;
	float lowerbound = 0.1;
	float scaling[SMS*parameters*TIME2];

	long map_dims[16];
	md_select_dims(16, FFT_FLAGS, map_dims, dims);
	long map_strs[16];
	md_calc_strides(16, map_strs, map_dims, CFL_SIZE);

	debug_printf(DP_DEBUG3, "##tau = %f\n", tau);

	for (itrdata.iter = 0; itrdata.iter < maxiter; itrdata.iter++) {

		iter_monitor(monitor, vops, x);

		ls_old = lambda_scale;
		lambda_scale = ist_continuation(&itrdata, continuation);
		
		if (lambda_scale != ls_old)
			debug_printf(DP_DEBUG3, "##lambda_scale = %f\n", lambda_scale);

		// normalize all the maps before joint wavelet denoising

		for (w = 0; w < TIME2; w++) {

			for (u = 0; u < SMS; u++) {

				for (v = 0; v < parameters; v++) {

					temp_index = v + u * parameters + w * SMS * parameters;
					scaling[temp_index] = md_norm(1, MD_DIMS(2 * md_calc_size(16, map_dims)), x + res * res * 2 * temp_index);
					md_smul(1, MD_DIMS(2*md_calc_size(16, map_dims)), x + res * res * 2 * temp_index,
					        x + res * res * 2 * temp_index, 1.0 / scaling[temp_index]);
				}
			}
		}

		iter_op_p_call(thresh, lambda_scale * tau, x, x);

		for (w = 0; w < TIME2; w++) {

			for (u = 0; u < SMS; u++) {

				for(v = 0; v < parameters; v++) {

					temp_index = v + u * parameters + w * SMS * parameters;
					md_smul(1, MD_DIMS(2 * md_calc_size(16, map_dims)), x + res * res * 2 * temp_index,
					        x + res * res * 2 * temp_index, scaling[temp_index]);
				}
			}
		}

		// Domain Prjoection for R1s
		for (w = 0; w < TIME2; w++) {

			for (u = 0; u < SMS; u++) {

				temp_index = res * res * 2 * (parameters-1) + (u + w * SMS) * res * res * 2 * parameters;
				vops->zsmax(md_calc_size(16, map_dims), (complex float)lowerbound, (complex float*)(x + temp_index), (complex float*)(x + temp_index));
			}
		}

		ravine(vops, N, &ra, x, o);	// FISTA
		iter_op_call(op, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		itrdata.rsnew = vops->norm(N, r);

		debug_printf(DP_DEBUG3, "#It %03d: %f   \n", itrdata.iter, itrdata.rsnew / itrdata.rsnot);

		if (itrdata.rsnew < epsilon)
			break;

		vops->axpy(N, x, tau, r);

		for (w = 0; w < TIME2; w++) {

			for (u = 0; u < SMS; u++) {

				temp_index = res * res * 2 * (parameters-1) + (u + w * SMS) * res * res * 2 * parameters;
				vops->zsmax(md_calc_size(16, map_dims), (complex float)lowerbound, (complex float*)(x + temp_index), (complex float*)(x + temp_index));
//				md_zreal(1, MD_DIMS(md_calc_size(16, map_dims)), x + temp_index, x + temp_index);
			}
		}


		if (hogwild)
			hogwild_k++;
		
		if (hogwild_k == hogwild_K) {

			hogwild_K *= 2;
			hogwild_k = 0;
			tau /= 2;
		}
	}

	debug_printf(DP_DEBUG3, "\n");
	debug_printf(DP_DEBUG2, "\t\tFISTA iterations: %u\n", itrdata.iter);

	vops->del(o);
	vops->del(r);
}



