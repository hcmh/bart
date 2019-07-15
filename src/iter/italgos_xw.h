/* Copyright 2013-2017. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifndef __ITALGOS_XW_H
#define __ITALGOS_XW_H

#include "misc/cppwrap.h"

#ifndef NUM_INTERNAL
// #warning "Use of private interfaces"
#endif

#include "misc/types.h"

struct vec_iter_s;
struct iter_op_s;
struct iter_nlop_s;
struct iter_op_p_s;
struct iter_monitor_s;

void fista_xw(unsigned int maxiter, float epsilon, float tau, long* dims,
	float continuation, _Bool hogwild,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor);

#include "misc/cppwrap.h"

#endif // __ITALGOS_XW_H


