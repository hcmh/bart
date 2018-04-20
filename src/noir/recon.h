/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

#include "misc/mri.h"


struct noir_conf_s {

	unsigned int iter;
	_Bool rvc;
	_Bool usegpu;
	_Bool noncart;
	_Bool nlinv_legacy;
	float alpha;
	float redu;
	float a;
	float b;
	float cgtol;
	_Bool pattern_for_each_coil;
	_Bool out_im_steps;
	_Bool out_coils_steps;
	_Complex float* out_im;
	_Complex float* out_coils;
};

extern const struct noir_conf_s noir_defaults;

extern void noir_recon(const struct noir_conf_s* conf, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* ref, const _Complex float* pattern, const _Complex float* mask, const _Complex float* kspace_data);

#include "misc/cppwrap.h"

