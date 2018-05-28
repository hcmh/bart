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
	float alpha;
	float alpha_min;
	float redu;
	float a;
	float b;
	_Bool pattern_for_each_coil;
	_Bool sms;
};

struct ds_s {

	long dims_full[DIMS];
	long dims_singleFrame[DIMS];
	long dims_singlePart[DIMS];
	long dims_singleFramePart[DIMS];
	long dims_output[DIMS];

	long strs_full[DIMS];
	long strs_singleFrame[DIMS];
	long strs_singlePart[DIMS];
	long strs_singleFramePart[DIMS];
	long strs_output[DIMS];
};

extern const struct noir_conf_s noir_defaults;

extern void noir_recon(const struct noir_conf_s* conf, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* ref, const _Complex float* pattern, const _Complex float* mask, const _Complex float* kspace_data);

extern void ds_init(struct ds_s* dims, size_t size);

extern void scale_psf_k(struct ds_s* pat_s, complex float* pattern, struct ds_s* k_s, complex float* kspace_data, struct ds_s* traj_s, complex float* traj);
#include "misc/cppwrap.h"

