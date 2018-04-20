/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 


#include "misc/mri.h"

struct linop_s;
extern void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src);
extern void noir_back_coils(const struct linop_s* op, complex float* dst, const complex float* src);

struct noir_model_conf_s {

	unsigned int fft_flags;
	unsigned int iter;
	_Bool rvc;
	_Bool use_gpu;
	_Bool noncart;
	float a;
	float b;
	_Bool pattern_for_each_coil;
	_Bool out_im_steps;
	_Bool out_coils_steps;
	_Complex float* out_im;
	_Complex float* out_coils;
};

extern struct noir_model_conf_s noir_model_conf_defaults;

struct nlop_s;

struct noir_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
	struct noir_op_s* noir_op;
};

extern struct noir_s noir_create2(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf);
extern struct noir_s noir_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf);

struct nlop_data_s;
extern void noir_dump(struct noir_s* op, const complex float* img, const complex float* coils);
extern void noir_orthogonalize(struct noir_s* op, complex float* coils);
