/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include "misc/cppwrap.h"

struct operator_p_s;
struct linop_s;

#ifndef DIMS
#define DIMS 16u
#endif

#ifndef NUM_REGS
#define NUM_REGS 10
#endif


struct opt_reg_s;


extern const struct operator_p_s* create_moba_nonneg_prox(unsigned int N, const long maps_dims[__VLA(N)], unsigned int coeff_dim, unsigned int coeff_flag, float lambda);


extern void help_reg_moba(void);

extern _Bool opt_reg_moba(void* ptr, char c, const char* optarg);

extern void opt_reg_moba_configure(unsigned int N, const long dims[__VLA(N)], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int model);
