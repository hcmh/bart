/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"
#include <stdint.h>

#define RAND_MAX_SPCG32 4294967295

extern double uniform_rand(void);
extern _Complex double gaussian_rand(void);
extern void md_gaussian_rand(unsigned int D, const long dims[__VLA(D)], _Complex float* dst);
extern void md_uniform_rand(unsigned int D, const long dims[__VLA(D)], _Complex float* dst);
extern void md_rand_one(unsigned int D, const long dims[__VLA(D)], _Complex float* dst, double p);
extern void num_rand_init(unsigned int seed);
extern uint32_t rand_spcg32(uint64_t s[1]);
extern _Complex double rand_spcg32_normal(uint64_t rand_seed[1]);



#include "misc/cppwrap.h"
