/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "rand.h"

unsigned int num_rand_seed = 123;


void num_rand_init(unsigned int seed)
{
	num_rand_seed = seed;
}


double uniform_rand(void)
{
	double ret;

	#pragma omp critical
	ret = rand_r(&num_rand_seed) / (double)RAND_MAX;

	return ret;
}


/**
 * Box-Muller
 */
complex double gaussian_rand(void)
{
	double u1, u2, s;

 	do {

		u1 = 2. * uniform_rand() - 1.;
		u2 = 2. * uniform_rand() - 1.;
   		s = u1 * u1 + u2 * u2;

   	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;
}

void md_gaussian_rand(unsigned int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_gaussian_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
//#pragma omp parallel for
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)gaussian_rand();
}

void md_uniform_rand(unsigned int D, const long dims[D], complex float* dst)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_uniform_rand(D, dims, tmp);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)uniform_rand();
}

void md_rand_one(unsigned int D, const long dims[D], complex float* dst, double p)
{
#ifdef  USE_CUDA
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(D, dims, sizeof(complex float));
		md_rand_one(D, dims, tmp, p);
		md_copy(D, dims, dst, tmp, sizeof(complex float));
		md_free(tmp);
		return;
	}
#endif
	for (long i = 0; i < md_calc_size(D, dims); i++)
		dst[i] = (complex float)(uniform_rand() < p);
}

// Generic random number generator
// adapted from Chris Wellons http://nullprogram.com/blog/2017/09/21/
uint32_t rand_spcg32(uint64_t s[1]) 
{
    uint64_t m = 0x9b60933458e17d7d;
    uint64_t a = 0xd737232eeccdf7ed;
    *s = *s * m + a;
    int shift = 29 - (*s >> 61);
    return (uint32_t)(*s >> shift);
}

// Generic normal-distributed random number generator 
complex double rand_spcg32_normal(uint64_t rand_seed[1])
{
	double u1, u2, s;

 	do {

		u1 = 2. * (rand_spcg32(rand_seed) / (1. * RAND_MAX_SPCG32)) - 1.;
		u2 = 2. * (rand_spcg32(rand_seed) / (1. * RAND_MAX_SPCG32))- 1.;
   		s = u1 * u1 + u2 * u2;

   	} while (s > 1.);

	double re = sqrt(-2. * log(s) / s) * u1;
	double im = sqrt(-2. * log(s) / s) * u2;

	return re + 1.i * im;
}