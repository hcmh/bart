/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */



#include "misc/mri.h"

struct linop_s;
struct nlop_s;


struct noir2_model_conf_s {

	_Bool noncart;

	unsigned int fft_flags_noncart;
	unsigned int fft_flags_cart;

	_Bool rvc;
	_Bool sos;
	float a;
	float b;

	struct nufft_conf_s* nufft_conf;
};

extern struct noir2_model_conf_s noir2_model_conf_defaults;


struct noir2_s {

	struct noir2_model_conf_s model_conf;
	struct nlop_s* nlop;
	const struct linop_s* lop_coil2;	// kspace coils to img-coils without resize
	const struct linop_s* lop_fft;		// fft/nufft from coil images to kspace
	const struct linop_s* lop_coil;		// kspace coils to img-coils
	const struct linop_s* lop_im;		// masking/resizing of image
	const struct nlop_s* tenmul;		// tenmul coil * image = coil image

	const struct linop_s* lop_nufft;	// for retrospectively changing trajectory
	const struct linop_s* lop_pattern;	// for retrospectively changing pattern

	int N;
	long* pat_dims;
	long* bas_dims;
	long* msk_dims;
	long* ksp_dims;
	long* cim_dims;
	long* img_dims;
	long* col_dims;
	long* trj_dims;
};

extern struct noir2_s noir2_noncart_create(int N,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf);

extern struct noir2_s noir2_cart_create(int N,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N],
	const struct noir2_model_conf_s* conf);

extern void noir2_free(struct noir2_s* model);

extern void noir2_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src);
extern void noir2_back_coils(const struct linop_s* op, complex float* dst, const complex float* src);
extern void noir2_orthogonalize(int N, const long col_dims[N], int idx, unsigned long flags, complex float* coils);