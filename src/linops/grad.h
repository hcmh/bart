/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

#ifndef __BC_ENUMS
#define __BC_ENUMS
enum BOUNDARY_CONDITION {BC_PERIODIC, BC_ZERO, BC_SAME};
#endif

extern struct linop_s* linop_grad_create(long N, const long dims[__VLA(N)], int d, unsigned int flags);
extern struct linop_s* linop_div_create(long N, const long dims[__VLA(N)], int d, unsigned int flags,
					const unsigned int order, const enum BOUNDARY_CONDITION bc);
extern struct linop_s *linop_fd_create(long N, const long dims[N], int d, unsigned int flags, unsigned int order,
				       const enum BOUNDARY_CONDITION bc, _Bool reverse);

#include "misc/cppwrap.h"
