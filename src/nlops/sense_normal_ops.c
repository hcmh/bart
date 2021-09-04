#include <math.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "noncart/nufft.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/rand.h"
#include "num/fft.h"

#include "iter/misc.h"
#include "iter/iter.h"
#include "iter/iter2.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/mri_ops.h"
#include "nlops/sense_normal_ops.h"


// this struct and the operators in it can be used to apply the normal eq operators and compute the regeularized inverse
struct sense_cart_normal_s {

	int N;

	unsigned long bat_flag;
	unsigned long img_flag;
	unsigned long col_flag;
	unsigned long cim_flag;
	unsigned long pat_flag;

	const long* bat_dims;
	const long* max_dims;
	const long* cim_dims;
	const long* img_dims;
	const long* col_dims;
	const long* pat_dims;

	const struct linop_s* lop_fft; //reusable
	const struct linop_s* lop_fft_mod; //reusable

	const struct operator_s** normal_ops;
};

static const struct operator_s* create_sense_cart_normal_op_int(struct sense_cart_normal_s* d, int N,
								const long col_dims[N], const complex float* coil,
								const long pat_dims[N], const complex float* pattern)
{
	assert(md_check_equal_dims(N, pat_dims, d->pat_dims, ~0));
	assert(md_check_equal_dims(N, col_dims, d->col_dims, ~0));

	long max_dims[d->N];
	md_select_dims(d->N, d->cim_flag | d->img_flag | d->col_flag, max_dims, d->max_dims);

	auto linop_frw = linop_chain_FF(linop_clone(d->lop_fft_mod), linop_fmac_create(N, max_dims, ~(d->cim_flag), ~(d->img_flag), ~(d->col_flag), coil));
	linop_frw = linop_chain_FF(linop_frw, linop_clone(d->lop_fft));

	auto linop_pattern = linop_cdiag_create(N, d->cim_dims, d->pat_flag, pattern);

	auto result = operator_chainN(3, (const struct operator_s **)MAKE_ARRAY(linop_frw->forward, linop_pattern->forward, linop_frw->adjoint));

	linop_free(linop_frw);
	linop_free(linop_pattern);

	return result;
}

static void sense_normal_cart_update_ops(	struct sense_cart_normal_s* d, int N,
						const long col_dims[N], const complex float* coil,
						const long pat_dims[N], const complex float* pattern)
{
	assert(N == d->N);

	long col_dims_slice[N];
	long pat_dims_slice[N];

	md_select_dims(N, ~d->bat_flag, col_dims_slice, col_dims);
	md_select_dims(N, ~d->bat_flag, pat_dims_slice, pat_dims);

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;

	do {
		int index = md_calc_offset(d->N, MD_STRIDES(d->N, d->bat_dims, 1), pos);
		if (NULL != (d->normal_ops)[index])
			operator_free((d->normal_ops)[index]);

		const complex float* coil_i = &MD_ACCESS(d->N, MD_STRIDES(N, col_dims, CFL_SIZE), pos, coil);
		const complex float* pattern_i = &MD_ACCESS(d->N, MD_STRIDES(N, pat_dims, CFL_SIZE), pos, pattern);

		(d->normal_ops)[index] = create_sense_cart_normal_op_int(d, N,
									col_dims_slice, coil_i,
									pat_dims_slice, pattern_i);

	} while (md_next(N, d->bat_dims, ~(0ul), pos));
}

static struct sense_cart_normal_s* sense_cart_normal_create(int N, const long max_dims[N], const long pat_dims[N], const struct config_nlop_mri_s* conf)
{
	assert(NULL != conf);
	assert(!conf->basis); // not implemented

	// batch dims must be outer most dims
	bool batch = false;
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(conf->batch_flags, i))
			batch = true;
		else
			assert(!batch || (1 == (max_dims)[i]));
	}

	PTR_ALLOC(struct sense_cart_normal_s, data);

	data->N = N;

	data->bat_flag = conf->batch_flags;
	data->img_flag = conf->image_flags;
	data->col_flag = conf->coil_flags;
	data->cim_flag = conf->coil_image_flags;
	data->pat_flag = conf->pattern_flags;

	PTR_ALLOC(long[N], n_bat_dims);
	PTR_ALLOC(long[N], n_max_dims);
	PTR_ALLOC(long[N], n_cim_dims);
	PTR_ALLOC(long[N], n_img_dims);
	PTR_ALLOC(long[N], n_col_dims);
	PTR_ALLOC(long[N], n_pat_dims);

	md_select_dims(N, data->bat_flag, *n_bat_dims, max_dims);
	md_select_dims(N, ~data->bat_flag, *n_max_dims, max_dims);
	md_select_dims(N, data->cim_flag & ~data->bat_flag, *n_cim_dims, max_dims);
	md_select_dims(N, data->img_flag & ~data->bat_flag, *n_img_dims, max_dims);
	md_select_dims(N, data->col_flag & ~data->bat_flag, *n_col_dims, max_dims);
	md_select_dims(N, data->pat_flag & ~data->bat_flag, *n_pat_dims, max_dims);

	assert(md_check_equal_dims(N, *n_pat_dims, pat_dims, ~(data->bat_flag)));


	data->bat_dims = *PTR_PASS(n_bat_dims);
	data->max_dims = *PTR_PASS(n_max_dims);
	data->cim_dims = *PTR_PASS(n_cim_dims);
	data->img_dims = *PTR_PASS(n_img_dims);
	data->col_dims = *PTR_PASS(n_col_dims);
	data->pat_dims = *PTR_PASS(n_pat_dims);

	PTR_ALLOC(const struct operator_s*[md_calc_size(N, data->bat_dims)], normalops);
	for (int i = 0; i < md_calc_size(N, data->bat_dims); i++)
		(*normalops)[i] = NULL;
	data->normal_ops = *PTR_PASS(normalops);

	unsigned long fft_flags = conf->fft_flags & md_nontriv_dims(N, pat_dims);

	data->lop_fft = linop_fft_create(N, data->cim_dims, fft_flags);

	// create linop for fftmod which only applies on coil dims not kdims
	long fft_dims[N];
	md_select_dims(N, fft_flags, fft_dims, data->img_dims);

	complex float* fmod = md_alloc(N, fft_dims, CFL_SIZE);
	md_zfill(N, fft_dims, fmod, 1.);
	fftmod(N, fft_dims, fft_flags, fmod, fmod);
	fftscale(N, fft_dims, fft_flags, fmod, fmod);

	data->lop_fft_mod = linop_cdiag_create(N, data->img_dims, fft_flags, fmod);
	md_free(fmod);

	return PTR_PASS(data);
}

static void sense_cart_normal_free(struct sense_cart_normal_s* d)
{
	for (int i = 0; i < md_calc_size(d->N, d->bat_dims); i++) {

		operator_free(d->normal_ops[i]);
		d->normal_ops[i] = NULL;
	}

	xfree(d->bat_dims);
	xfree(d->max_dims);
	xfree(d->cim_dims);
	xfree(d->img_dims);
	xfree(d->col_dims);
	xfree(d->pat_dims);

	linop_free(d->lop_fft);
	linop_free(d->lop_fft_mod);

	xfree(d->normal_ops);

	xfree(d);
}




// this struct and the operators in it can be used to apply the normal eq operators and compute the regeularized inverse
struct sense_noncart_normal_s {

	int N;
	int ND;

	unsigned long bat_flag;
	unsigned long img_flag;
	unsigned long col_flag;
	unsigned long cim_flag;

	const long* max_dims;
	const long* cim_dims;
	const long* psf_dims;

	const long* bat_dims;

	const struct operator_s** nufft_normal_ops;
	const struct operator_s** normal_ops;
};

static void sense_noncart_normal_release_ops(struct sense_noncart_normal_s* d)
{
	for (int i = 0; i < md_calc_size(d->N, d->bat_dims); i++) {

		operator_free(d->normal_ops[i]);
		d->normal_ops[i] = NULL;
	}
}

static void create_nufft_normal_ops(struct sense_noncart_normal_s* d, bool basis, struct nufft_conf_s conf)
{
	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;

	do {
		int index = md_calc_offset(d->N, MD_STRIDES(d->N, d->bat_dims, 1), pos);
		d->nufft_normal_ops[index] = nufft_normal_op_create(d->N, d->cim_dims, d->psf_dims, basis, conf);

	} while (md_next(d->N, d->bat_dims, ~(0ul), pos));
}

static void sense_normal_noncart_update_ops(	struct sense_noncart_normal_s* d, int N,
						const long col_dims[N], const complex float* coils,
						const long psf_dims[N + 1], const complex float* psf)
{
	long psf_strs[N + 1];
	md_calc_strides(N + 1, psf_strs, psf_dims, CFL_SIZE);

	long col_strs[N];
	md_calc_strides(N, col_strs, col_dims, CFL_SIZE);

	long pos[d->ND];
	for (int i = 0; i < d->ND; i++)
		pos[i] = 0;

	do {
		int index = md_calc_offset(d->N, MD_STRIDES(d->N, d->bat_dims, 1), pos);

		if (NULL != d->normal_ops[index])
			operator_free(d->normal_ops[index]);

		nufft_normal_op_set_psf2(d->nufft_normal_ops[index], d->N + 1, d->psf_dims, psf_strs, &MD_ACCESS(N + 1, psf_strs, pos, psf));

		long max_dims[d->N];
		md_select_dims(d->N, d->cim_flag | d->img_flag | d->col_flag, max_dims, d->max_dims);

		auto linop_fmac = linop_fmac_create(d->N, max_dims, ~d->cim_flag, ~d->img_flag, ~d->col_flag, &MD_ACCESS(N, col_strs, pos, coils));

		d->normal_ops[index] = operator_chainN(3, (const struct operator_s *[3]) { linop_fmac->forward, d->nufft_normal_ops[index], linop_fmac->adjoint });

		linop_free(linop_fmac);

	} while (md_next(d->N, d->bat_dims, ~(0ul), pos));
}

static struct sense_noncart_normal_s* sense_noncart_normal_create(int N, const long max_dims[N], const long psf_dims[N + 1], const struct config_nlop_mri_s* conf)
{
	assert(NULL != conf);

	PTR_ALLOC(struct sense_noncart_normal_s, data);

	// batch dims must be outer most dims (not including last psf dim)
	bool batch = false;
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(conf->batch_flags, i))
			batch = true;
		else
			assert(!batch || (1 == max_dims[i]));
	}

	data->N = N;
	data->ND = N +1;

	data->bat_flag = conf->batch_flags;
	data->col_flag = conf->coil_flags;
	data->img_flag = conf->image_flags;
	data->cim_flag = conf->coil_image_flags;

	PTR_ALLOC(long[N], n_max_dims);
	PTR_ALLOC(long[N], n_cim_dims);
	PTR_ALLOC(long[N], n_bat_dims);
	PTR_ALLOC(long[N + 1], n_psf_dims);

	md_select_dims(N, ~conf->batch_flags & conf->coil_image_flags, *n_cim_dims, max_dims);
	md_select_dims(N, ~conf->batch_flags, *n_max_dims, max_dims);
	md_select_dims(N, conf->batch_flags, *n_bat_dims, max_dims);

	md_select_dims(N + 1, ~conf->batch_flags, *n_psf_dims, psf_dims);

	data->cim_dims = *PTR_PASS(n_cim_dims);
	data->psf_dims = *PTR_PASS(n_psf_dims);
	data->bat_dims = *PTR_PASS(n_bat_dims);
	data->max_dims = *PTR_PASS(n_max_dims);

	PTR_ALLOC(const struct operator_s*[md_calc_size(N, data->bat_dims)], nufft_normal_ops);
	for (int i = 0; i < md_calc_size(N, data->bat_dims); i++)
		(*nufft_normal_ops)[i] = NULL;
	data->nufft_normal_ops = *PTR_PASS(nufft_normal_ops);

	create_nufft_normal_ops(data, conf->basis, *(conf->nufft_conf));


	PTR_ALLOC(const struct operator_s*[md_calc_size(N, data->bat_dims)], normalops);
	for (int i = 0; i < md_calc_size(N, data->bat_dims); i++)
		(*normalops)[i] = NULL;
	data->normal_ops = *PTR_PASS(normalops);

	return PTR_PASS(data);
}



static void sense_noncart_normal_ops_data_free(struct sense_noncart_normal_s* d)
{
	sense_noncart_normal_release_ops(d);
	for (int i = 0; i < md_calc_size(d->N, d->bat_dims); i++)
		operator_free(d->nufft_normal_ops[i]);

	xfree(d->nufft_normal_ops);
	xfree(d->normal_ops);

	xfree(d->max_dims);
	xfree(d->cim_dims);
	xfree(d->psf_dims);
	xfree(d->bat_dims);

	xfree(d);
}

struct sense_normal_ops_s* sense_normal_create(int N, const long max_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf)
{
	PTR_ALLOC(struct sense_normal_ops_s, data);

	PTR_ALLOC(long[ND], n_psf_dims);
	md_copy_dims(ND, *n_psf_dims, psf_dims);
	data->psf_dims = *PTR_PASS(n_psf_dims);

	PTR_ALLOC(long[N], img_dims);
	PTR_ALLOC(long[N], col_dims);
	PTR_ALLOC(long[N], bat_dims);
	PTR_ALLOC(long[N], img_dims_slice);

	md_select_dims(N, conf->image_flags, *img_dims, max_dims);
	md_select_dims(N, conf->coil_flags, *col_dims, max_dims);
	md_select_dims(N, conf->batch_flags, *bat_dims, max_dims);
	md_select_dims(N, conf->image_flags & (~conf->batch_flags), *img_dims_slice, max_dims);

	data->N = N;
	data->ND = ND;

	data->img_dims = *PTR_PASS(img_dims);
	data->col_dims = *PTR_PASS(col_dims);
	data->bat_dims = *PTR_PASS(bat_dims);
	data->img_dims_slice = *PTR_PASS(img_dims_slice);

	data->bat_flag = conf->batch_flags;
	data->img_flag = conf->image_flags;

	data->noncart = conf->noncart;

	data->sense_noncart = NULL;
	data->sense_cart = NULL;

	if (data->noncart) {

		assert(N + 1 == ND);
		data->sense_noncart = sense_noncart_normal_create(N, max_dims, psf_dims, conf);
	} else {

		assert(N == ND);
		data->sense_cart = sense_cart_normal_create(N, max_dims, psf_dims, conf);
	}

	return PTR_PASS(data);
}

void sense_normal_free(struct sense_normal_ops_s* d)
{
	if (NULL != d->sense_noncart)
		sense_noncart_normal_ops_data_free(d->sense_noncart);
	if (NULL != d->sense_cart)
		sense_cart_normal_free(d->sense_cart);

	xfree(d->psf_dims);
	xfree(d->col_dims);

	xfree(d->img_dims);
	xfree(d->img_dims_slice);
	xfree(d->bat_dims);

	xfree(d);
}

void sense_normal_update_ops(	struct sense_normal_ops_s* d,
				int N, const long col_dims[N], const complex float* coils,
				int ND, const long psf_dims[ND], const complex float* psf)
{
	if (NULL != d->sense_noncart) {

		assert(N + 1 == ND);

		sense_normal_noncart_update_ops(d->sense_noncart, N, col_dims, coils, psf_dims, psf);
	}

	if (NULL != d->sense_cart) {

		assert(N == ND);

		sense_normal_cart_update_ops(d->sense_cart, N, col_dims, coils, psf_dims, psf);
	}
}

const struct operator_s* sense_get_normal_op(struct sense_normal_ops_s* d, int N, const long pos[N])
{
	assert(!(md_nontriv_strides(N, pos) & (!md_nontriv_dims(N, d->bat_dims))));
	int index = md_calc_offset(d->N, MD_STRIDES(d->N, d->bat_dims, 1), pos);

	if (NULL != d->sense_noncart)
		return d->sense_noncart->normal_ops[index];

	if (NULL != d->sense_cart)
		return d->sense_cart->normal_ops[index];

	assert(0);
	return NULL;
}

void sense_apply_normal_ops(struct sense_normal_ops_s* d, int N, const long img_dims[N], complex float* dst, const complex float* src)
{

	assert(N == d->N);

	long img_strs[N];
	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;

	do {
		const struct operator_s* normal_op = sense_get_normal_op(d, N, pos);
		operator_apply(	normal_op,
				d->N, d->img_dims_slice, &MD_ACCESS(d->N, img_strs, pos, dst),
				d->N, d->img_dims_slice, &MD_ACCESS(d->N, img_strs, pos, src)
				);

	} while (md_next(d->N, d->bat_dims, ~(0ul), pos));
}

void sense_apply_normal_inv(struct sense_normal_ops_s* d, const struct iter_conjgrad_conf* iter_conf, int N, const long img_dims[N], complex float* dst, const complex float* src)
{

	assert(N == d->N);

	long img_strs[N];
	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

	long img_dims_slice[N];
	md_select_dims(N, ~d->bat_flag, img_dims_slice, img_dims);

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;

	do {
		const struct operator_s* normal_op = sense_get_normal_op(d, N, pos);

		int index = md_calc_offset(N, MD_STRIDES(N, d->bat_dims, 1), pos);

		md_clear(N, img_dims_slice, &MD_ACCESS(N, img_strs, pos, dst), CFL_SIZE);

		iter2_conjgrad(	CAST_UP(&(iter_conf[index])), normal_op,
				0, NULL, NULL, NULL, NULL,
				2 * md_calc_size(N, img_dims_slice),
				(float*)&MD_ACCESS(N, img_strs, pos, dst),
				(const float*)&MD_ACCESS(N, img_strs, pos, src),
				NULL);

	} while (md_next(d->N, d->bat_dims, ~(0ul), pos));
}


