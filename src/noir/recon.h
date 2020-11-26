/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

#include "misc/mri.h"
#include "optreg.h"

struct noir_conf_s {

	unsigned int iter;
	_Bool rvc;
	_Bool noncart;
	float alpha;
	float alpha_min;
	float redu;
	float a;
	float b;
	unsigned int inner_iter;
	_Bool pattern_for_each_coil;
	_Bool sms;
	unsigned int cnstcoil_flags;
	_Bool img_space_coils;
	float step;
	unsigned int algo;
	float rho;
	struct opt_reg_s ropts;
	float tol;
	unsigned int shift_mode;

};

extern const struct noir_conf_s noir_defaults;

extern void noir_recon(const struct noir_conf_s* conf,
		       const long dims[DIMS],
		       _Complex float* img,
		       _Complex float* sens,
		       _Complex float* ksens,
		       const _Complex float* ref,
		       const _Complex float* pattern,
		       const _Complex float* mask,
		       const _Complex float* kspace_data);


#include "misc/cppwrap.h"

