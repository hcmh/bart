/* Copyright 2017-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/types.h"

struct operator_p_s;
struct iter3_conf_s;
struct iter_op_s;
struct iter_nlop_s;
struct nlop_s;
struct linop_s;

typedef void iter4_lop_fun_f(const struct iter3_conf_s* _conf,
		struct nlop_s* nlop,
		struct linop_s* lop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* solve,
		const struct iter_op_s cb);

iter4_lop_fun_f iter4_lop_irgnm;
iter4_lop_fun_f iter4_lop_irgnm2;
