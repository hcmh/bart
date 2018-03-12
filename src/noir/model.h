/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 


#include "misc/mri.h"

struct nlop_s;
extern void noir_forw_coils(struct nlop_s* data, complex float* dst, const complex float* src);
extern void noir_back_coils(struct nlop_s* data, complex float* dst, const complex float* src);

struct noir_model_conf_s {

	unsigned int fft_flags;
	_Bool rvc;
	_Bool use_gpu;
	_Bool noncart;
};

extern struct noir_model_conf_s noir_model_conf_defaults;

extern struct nlop_s* noir_create2(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf);
extern struct nlop_s* noir_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf);


