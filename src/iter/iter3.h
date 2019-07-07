/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/types.h"

typedef struct iter3_conf_s { TYPEID* TYPEID; } iter3_conf;

struct iter_op_s;


extern void iter3_irgnm(iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src);


extern void iter3_irgnm_l1(iter3_conf* _conf,
		struct iter_op_s frw,
		struct iter_op_s der,
		struct iter_op_s adj,
		long N, float* dst, const float* ref,
		long M, const float* src);


struct iter3_irgnm_conf {

	INTERFACE(iter3_conf);

	int iter;
	float alpha;
	float alpha_min;
	float redu;

	int cgiter;
	float cgtol;
    
	long N;
	long dims[16];

	_Bool nlinv_legacy;
};



struct iter3_landweber_conf {

	INTERFACE(iter3_conf);

	int iter;
	float alpha;
	float epsilon;
};




extern const struct iter3_irgnm_conf iter3_irgnm_defaults;
extern const struct iter3_landweber_conf iter3_landweber_defaults;

