#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "num/fft.h"
#include "num/filter.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "simu/signals.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"

#include "meco.h"
#include "noir/model.h"


#define FATPEAKS 6


struct meco_s {

	INTERFACE(nlop_data_t);

	int N;
	long model;

	const long* y_dims;
	const long* x_dims;
	const long* der_dims;
	const long* map_dims;
	const long* TE_dims;
	
	const long* y_strs;
	const long* x_strs;
	const long* der_strs;
	const long* map_strs;
	const long* TE_strs;

	// Parameter maps
	complex float* der_x;
	complex float* TE;
	complex float* cshift;
	complex float* scaling; // length = number of maps
	complex float* weights;

	const struct linop_s* linop_fB0;
};

DEF_TYPEID(meco_s);


long set_num_of_coeff(unsigned int sel_model)
{
	long ncoeff = 0;

	switch (sel_model) {
		case WF     : ncoeff = 3; break;
		case WFR2S  : ncoeff = 4; break;
		case WF2R2S : ncoeff = 5; break;
		case R2S    : ncoeff = 3; break;
	}

	return ncoeff;
}



void meco_calc_fat_modu(unsigned int N, const long dims[N], const complex float* TE, complex float* dst)
{
	struct signal_model meco = signal_multi_grad_echo_defaults;
	meco.delta_b0 = 3.0;

	md_clear(N, dims, dst, CFL_SIZE);

	for (int eind = 0; eind < dims[TE_DIM]; eind++) {

		meco.te = TE[eind];
		multi_grad_echo_model(&meco, 1, &dst[eind]);
	}
}



static void meco_calc_weights(const long dims[3], complex float* dst, int weight_type)
{
	unsigned int flags = 0;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);

	switch (weight_type) {
		case 0: 
			md_clear(3, dims, dst, CFL_SIZE);
			md_zsadd(3, dims, dst, dst, 1.);
			break;
		
		case 1: 
			klaplace(3, dims, flags, dst);
			md_zsmul(3, dims, dst, dst, 44.);
			md_zsadd(3, dims, dst, dst, 1.);
			md_zspow(3, dims, dst, dst, -16.);
			break;
	}
}

const struct linop_s* meco_get_fB0_trafo(struct nlop_s* op)
{
	const nlop_data_t* _data = nlop_get_data(op);
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	return data->linop_fB0;
}

void meco_forw_fB0(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void meco_back_fB0(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}

// ************************************************************* //
//  Model: (W + F cshift) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wf(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	long* pos = calloc(data->N, sizeof(long));
	enum { PIND_W, PIND_F, PIND_FB0 };
	int m;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //
	// W
	pos[COEFF_DIM] = PIND_W;
	const complex float* W = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	// F
	pos[COEFF_DIM] = PIND_F;
	const complex float* F = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)dst + data->y_strs[TE_DIM] * m, data->map_strs, F, data->cshift[m] * data->scaling[PIND_F]);
	}

	// dst = W + F .* cshift
	md_zadd2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, dst, data->map_strs, W);

	// fB0
	pos[COEFF_DIM] = PIND_FB0;
	const complex float* fB0 = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);
	meco_forw_fB0(data->linop_fB0, tmp_map, fB0);

	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)tmp_exp + data->y_strs[TE_DIM] * m, data->map_strs, tmp_map, I*2.*M_PI*data->TE[m] * data->scaling[PIND_FB0]);
	}

	// tmp_exp = exp(1i*2*pi * fB0 .* TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);

	// dst = dst .* tmp_exp
	md_zmul(data->N, data->y_dims, dst, dst, tmp_exp);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	pos[COEFF_DIM] = PIND_W;
	complex float* der_W = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_copy(data->N, data->y_dims, der_W, tmp_exp, CFL_SIZE);

	// der_F
	pos[COEFF_DIM] = PIND_F;
	complex float* der_F = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_F + data->y_strs[TE_DIM] * m, data->y_strs, (void*)tmp_exp + data->y_strs[TE_DIM] * m, data->scaling[PIND_F] * data->cshift[m]);
	}

	// der_fB0
	pos[COEFF_DIM] = PIND_FB0;
	complex float* der_fB0 = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_fB0 + data->y_strs[TE_DIM] * m, data->y_strs, (void*)dst + data->y_strs[TE_DIM] * m, I*2.*M_PI*data->TE[m] * data->scaling[PIND_FB0]);
	}

	md_free(tmp_map);
	md_free(tmp_exp);
	xfree(pos);
}


// ************************************************************* //
//  Model: (W + F cshift) .* exp(- R2s TE) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wfr2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	long* pos = calloc(data->N, sizeof(long));
	enum { PIND_W, PIND_F, PIND_R2S, PIND_FB0 };
	int m;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_map1 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);    

	// =============================== //
	//  forward operator
	// =============================== //
	// W
	pos[COEFF_DIM] = PIND_W;
	const complex float* W = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	// F
	pos[COEFF_DIM] = PIND_F;
	const complex float* F = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)dst + data->y_strs[TE_DIM] * m, data->map_strs, F, data->scaling[PIND_F] * data->cshift[m]);
	}

	// dst = W + F .* cshift
	md_zadd2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, dst, data->map_strs, W);

	// R2s and fB0
	pos[COEFF_DIM] = PIND_R2S;
	const complex float* R2s = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, tmp_map, data->map_strs, R2s, -1.*data->scaling[PIND_R2S]);
	
	pos[COEFF_DIM] = PIND_FB0;
	const complex float* fB0 = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);
	meco_forw_fB0(data->linop_fB0, tmp_map1, fB0);
	md_zaxpy2(data->N, data->map_dims, data->map_strs, tmp_map, I*2.*M_PI*data->scaling[PIND_FB0], data->map_strs, tmp_map1);

	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)tmp_exp + data->y_strs[TE_DIM] * m, data->map_strs, tmp_map, data->TE[m]);
	}

	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);

	// dst = dst .* tmp_exp
	md_zmul(data->N, data->y_dims, dst, dst, tmp_exp);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	pos[COEFF_DIM] = PIND_W;
	complex float* der_W = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_copy(data->N, data->y_dims, der_W, tmp_exp, CFL_SIZE);

	// der_F
	pos[COEFF_DIM] = PIND_F;
	complex float* der_F = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_F + data->y_strs[TE_DIM] * m, data->y_strs, (void*)tmp_exp + data->y_strs[TE_DIM] * m, data->scaling[PIND_F] * data->cshift[m]);
	}

	// der_R2s
	pos[COEFF_DIM] = PIND_R2S;
	complex float* der_R2s = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_R2s + data->y_strs[TE_DIM] * m, data->y_strs, (void*)dst + data->y_strs[TE_DIM] * m, -1. * data->scaling[PIND_R2S] * data->TE[m]);
	}

	// der_fB0
	pos[COEFF_DIM] = PIND_FB0;
	complex float* der_fB0 = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_fB0 + data->y_strs[TE_DIM] * m, data->y_strs, (void*)dst + data->y_strs[TE_DIM] * m, I*2.*M_PI*data->TE[m] * data->scaling[PIND_FB0]);
	}

	md_free(tmp_map);
	md_free(tmp_map1);
	md_free(tmp_exp);
	xfree(pos);
}

// ************************************************************* //
//  Model: (W exp(- R2s_W TE) + F cshift exp(- R2s_F TE)) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wf2r2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	long* pos = calloc(data->N, sizeof(long));
	enum { PIND_W, PIND_R2SW, PIND_F, PIND_R2SF, PIND_FB0 };
	int m;

	complex float* tmp_exp_fB0  = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_exp_R2sW = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_exp_R2sF = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco      = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_map      = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //
	// W
	pos[COEFF_DIM] = PIND_W;
	const complex float* W = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	// R2sW
	pos[COEFF_DIM] = PIND_R2SW;
	const complex float* R2sW = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	// F
	pos[COEFF_DIM] = PIND_F;
	const complex float* F = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	// R2sF
	pos[COEFF_DIM] = PIND_R2SF;
	const complex float* R2sF = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	// fB0
	pos[COEFF_DIM] = PIND_FB0;
	const complex float* fB0 = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);
	meco_forw_fB0(data->linop_fB0, tmp_map, fB0);

	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)tmp_exp_R2sW + data->y_strs[TE_DIM] * m, data->map_strs, R2sW, -1.*data->TE[m] * data->scaling[PIND_R2SW]);

		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)tmp_eco + data->y_strs[TE_DIM] * m, data->map_strs, F, data->cshift[m] * data->scaling[PIND_F]);

		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)tmp_exp_R2sF + data->y_strs[TE_DIM] * m, data->map_strs, R2sF, -1.*data->TE[m] * data->scaling[PIND_R2SF]);

		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)tmp_exp_fB0 + data->y_strs[TE_DIM] * m, data->map_strs, tmp_map, I*2.*M_PI*data->TE[m] * data->scaling[PIND_FB0]);
	}

	// tmp_exp_R2sW = exp(- R2sW TE)
	md_zexp(data->N, data->y_dims, tmp_exp_R2sW, tmp_exp_R2sW);

	// tmp_exp_R2sF = exp(- R2sF TE)
	md_zexp(data->N, data->y_dims, tmp_exp_R2sF, tmp_exp_R2sF);

	// tmp_exp_fB0 = exp(i 2\pi fB0 TE)
	md_zexp(data->N, data->y_dims, tmp_exp_fB0, tmp_exp_fB0);

	// tmp_eco = W exp(- R2s_W TE) + F cshift exp(- R2s_F TE)
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_eco, tmp_exp_R2sF);
	md_zfmac2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, W, data->y_strs, tmp_exp_R2sW);

	// dst = tmp_eco .* tmp_exp_fB0
	md_zmul(data->N, data->y_dims, dst, tmp_eco, tmp_exp_fB0);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	pos[COEFF_DIM] = PIND_W;
	complex float* der_W = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_zmul(data->N, data->y_dims, der_W, tmp_exp_fB0, tmp_exp_R2sW);

	// der_R2sW
	pos[COEFF_DIM] = PIND_R2SW;
	complex float* der_R2sW = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_zmul2(data->N, data->y_dims, data->y_strs, der_R2sW, data->map_strs, W, data->y_strs, der_W);

	// der_F
	pos[COEFF_DIM] = PIND_F;
	complex float* der_F = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_zmul(data->N, data->y_dims, der_F, tmp_exp_fB0, tmp_exp_R2sF);

	// der_R2sF
	pos[COEFF_DIM] = PIND_R2SF;
	complex float* der_R2sF = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_zmul2(data->N, data->y_dims, data->y_strs, der_R2sF, data->map_strs, F, data->y_strs, der_F);

	// der_fB0
	pos[COEFF_DIM] = PIND_FB0;
	complex float* der_fB0 = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_zmul(data->N, data->y_dims, der_fB0, tmp_exp_fB0, tmp_eco);

	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_R2sW + data->y_strs[TE_DIM] * m, data->y_strs, (void*)der_R2sW + data->y_strs[TE_DIM] * m, -1.*data->TE[m] * data->scaling[PIND_R2SW]);

		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_F + data->y_strs[TE_DIM] * m, data->y_strs, (void*)der_F + data->y_strs[TE_DIM] * m, data->cshift[m] * data->scaling[PIND_F]);

		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_R2sF + data->y_strs[TE_DIM] * m, data->y_strs, (void*)der_R2sF + data->y_strs[TE_DIM] * m, -1.*data->TE[m]*data->cshift[m] * data->scaling[PIND_R2SF]);

		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_fB0 + data->y_strs[TE_DIM] * m, data->y_strs, (void*)der_fB0 + data->y_strs[TE_DIM] * m, I*2.*M_PI*data->TE[m] * data->scaling[PIND_FB0]);
	}

	md_free(tmp_exp_fB0);
	md_free(tmp_exp_R2sW);
	md_free(tmp_exp_R2sF);
	md_free(tmp_eco);
	md_free(tmp_map);
	xfree(pos);
}


// ************************************************************* //
//  Model: rho .* exp(- R2s TE) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_r2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	long* pos = calloc(data->N, sizeof(long));
	enum { PIND_RHO, PIND_R2S, PIND_FB0 };
	int m;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_map1 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //
	// rho
	pos[COEFF_DIM] = PIND_RHO;
	const complex float* rho = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);

	// R2s and fB0
	pos[COEFF_DIM] = PIND_R2S;
	const complex float* R2s = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, tmp_map, data->map_strs, R2s, -1.*data->scaling[PIND_R2S]);
	
	pos[COEFF_DIM] = PIND_FB0;
	const complex float* fB0 = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);
	meco_forw_fB0(data->linop_fB0, tmp_map1, fB0);
	md_zaxpy2(data->N, data->map_dims, data->map_strs, tmp_map, I*2.*M_PI*data->scaling[PIND_FB0], data->map_strs, tmp_map1);

	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)tmp_exp + data->y_strs[TE_DIM] * m, data->map_strs, tmp_map, data->TE[m]);
	}

	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);

	// dst = tmp_exp .* rho
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, tmp_exp, data->map_strs, rho);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_rho
	pos[COEFF_DIM] = PIND_RHO;
	complex float* der_rho = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	md_copy(data->N, data->y_dims, der_rho, tmp_exp, CFL_SIZE);

	// der_R2s
	pos[COEFF_DIM] = PIND_R2S;
	complex float* der_R2s = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_R2s + data->y_strs[TE_DIM] * m, data->y_strs, (void*)dst + data->y_strs[TE_DIM] * m, -1. * data->scaling[PIND_R2S] * data->TE[m]);
	}

	// der_fB0
	pos[COEFF_DIM] = PIND_FB0;
	complex float* der_fB0 = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);
	for (m = 0; m < data->TE_dims[TE_DIM]; m++) {
		md_zsmul2(data->N, data->map_dims, data->y_strs, (void*)der_fB0 + data->y_strs[TE_DIM] * m, data->y_strs, (void*)dst + data->y_strs[TE_DIM] * m, I*2.*M_PI * data->scaling[PIND_FB0] * data->TE[m]);
	}

	md_free(tmp_map);
	md_free(tmp_map1);
	md_free(tmp_exp);
	xfree(pos);
}


static void meco_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	long* pos = calloc(data->N, sizeof(long));

	complex float* tmp_fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_clear(data->N, data->y_dims, dst, CFL_SIZE);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {
		
		pos[COEFF_DIM] = pind;

		const complex float* tmp_map = (const void*)src + md_calc_offset(data->N, data->x_strs, pos);
		complex float* tmp_exp = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);

		if (pind == data->x_dims[COEFF_DIM]-1) {
			meco_forw_fB0(data->linop_fB0, tmp_fB0, tmp_map);
			md_zfmac2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, tmp_fB0, data->y_strs, tmp_exp);
		} else {
			md_zfmac2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, tmp_map, data->y_strs, tmp_exp);
		}

	}

	md_free(tmp_fB0);
	xfree(pos);
}

static void meco_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	long* pos = calloc(data->N, sizeof(long));

	md_clear(data->N, data->x_dims, dst, CFL_SIZE);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {
		
		pos[COEFF_DIM] = pind;

		complex float* tmp_map = (void*)dst + md_calc_offset(data->N, data->x_strs, pos);
		complex float* tmp_exp = (void*)data->der_x + md_calc_offset(data->N, data->der_strs, pos);

		md_zfmacc2(data->N, data->y_dims, data->map_strs, tmp_map, data->y_strs, src, data->y_strs, tmp_exp);
	}

	// real constraint on fB0
	pos[COEFF_DIM] = data->x_dims[COEFF_DIM] - 1;
	complex float* tmp_map = (void*)dst + md_calc_offset(data->N, data->x_strs, pos);
#if 1
	md_zreal(data->N, data->map_dims, tmp_map, tmp_map);
#endif
	meco_back_fB0(data->linop_fB0, tmp_map, tmp_map);

	// real constraint on R2S
#if 1
	if ((data->model == WFR2S) || (data->model == R2S)) {

		pos[COEFF_DIM] = data->x_dims[COEFF_DIM] - 2;
		tmp_map = (void*)dst + md_calc_offset(data->N, data->x_strs, pos);
		md_zreal(data->N, data->map_dims, tmp_map, tmp_map);

	} else
	if ( data->model == WF2R2S ) {

		pos[COEFF_DIM] = 3;
		tmp_map = (void*)dst + md_calc_offset(data->N, data->x_strs, pos);
		md_zreal(data->N, data->map_dims, tmp_map, tmp_map);

		pos[COEFF_DIM] = 1;
		tmp_map = (void*)dst + md_calc_offset(data->N, data->x_strs, pos);
		md_zreal(data->N, data->map_dims, tmp_map, tmp_map);

	}
#endif

	xfree(pos);
}

static void meco_del(const nlop_data_t* _data)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	md_free(data->TE);
	md_free(data->cshift);
	md_free(data->scaling);
	md_free(data->weights);

	md_free(data->der_x);

	xfree(data->y_dims);
	xfree(data->x_dims);
	xfree(data->der_dims);
	xfree(data->map_dims);
	xfree(data->TE_dims);

	xfree(data->y_strs);
	xfree(data->x_strs);
	xfree(data->der_strs);
	xfree(data->map_strs);
	xfree(data->TE_strs);

	linop_free(data->linop_fB0);

	xfree(data);
}


struct nlop_s* nlop_meco_create(const int N, const long y_dims[N], const long x_dims[N], const complex float* TE, unsigned int sel_model, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif
	
	PTR_ALLOC(struct meco_s, data);
	SET_TYPEID(meco_s, data);


	PTR_ALLOC(long[N], nydims);
	md_copy_dims(N, *nydims, y_dims);
	data->y_dims = *PTR_PASS(nydims);

	assert(x_dims[COEFF_DIM] == set_num_of_coeff(sel_model));
	data->model = sel_model;

	PTR_ALLOC(long[N], nxdims);
	md_copy_dims(N, *nxdims, x_dims);
	data->x_dims = *PTR_PASS(nxdims);

	PTR_ALLOC(long[N], nderdims);
	md_merge_dims(N, *nderdims, y_dims, x_dims);
	data->der_dims = *PTR_PASS(nderdims);

	long map_dims[N];
	md_select_dims(N, ~COEFF_FLAG, map_dims, x_dims);
	PTR_ALLOC(long[N], n1dims);
	md_copy_dims(N, *n1dims, map_dims);
	data->map_dims = *PTR_PASS(n1dims);

	long TE_dims[N];
	md_select_dims(N, TE_FLAG, TE_dims, y_dims);
	PTR_ALLOC(long[N], ntedims);
	md_copy_dims(N, *ntedims, TE_dims);
	data->TE_dims = *PTR_PASS(ntedims);

	long scaling_dims[N];
	md_select_dims(N, COEFF_FLAG, scaling_dims, x_dims);

	
	PTR_ALLOC(long[N], nystr);
	md_calc_strides(N, *nystr, y_dims, CFL_SIZE);
	data->y_strs = *PTR_PASS(nystr);

	PTR_ALLOC(long[N], nxstr);
	md_calc_strides(N, *nxstr, x_dims, CFL_SIZE);
	data->x_strs = *PTR_PASS(nxstr);

	PTR_ALLOC(long[N], nderstr);
	md_calc_strides(N, *nderstr, data->der_dims, CFL_SIZE);
	data->der_strs = *PTR_PASS(nderstr);

	PTR_ALLOC(long[N], n1str);
	md_calc_strides(N, *n1str, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(n1str);

	PTR_ALLOC(long[N], ntestr);
	md_calc_strides(N, *ntestr, TE_dims, CFL_SIZE);
	data->TE_strs = *PTR_PASS(ntestr);
	
	data->N = N;
	data->der_x = my_alloc(N, data->der_dims, CFL_SIZE);
	
	// echo times
	data->TE = md_alloc(N, TE_dims, CFL_SIZE);
	md_copy(N, TE_dims, data->TE, TE, CFL_SIZE);


	// calculate cshift
	data->cshift = md_alloc(N, TE_dims, CFL_SIZE);
	meco_calc_fat_modu(N, TE_dims, data->TE, data->cshift);


	// weight on fB0
	long w_dims[N];
	md_select_dims(N, FFT_FLAGS, w_dims, data->x_dims);

	data->weights = md_alloc(N, w_dims, CFL_SIZE);
	meco_calc_weights(w_dims, data->weights, 1);

	const struct linop_s* linop_wghts = linop_cdiag_create(N, data->map_dims, FFT_FLAGS, data->weights);
	const struct linop_s* linop_ifftc = linop_ifftc_create(N, data->map_dims, FFT_FLAGS);

	data->linop_fB0 = linop_chain(linop_wghts, linop_ifftc);

	linop_free(linop_wghts);
	linop_free(linop_ifftc);


	// scaling
	complex float* scaling = calloc(x_dims[COEFF_DIM], CFL_SIZE);

	for (int pind = 0; pind < x_dims[COEFF_DIM]; pind++) {
		scaling[pind] = 1.0 + 0.0 * I;
	}

	nlop_fun_t meco_fun = meco_fun_wf;
	switch (sel_model) {
		case WF: break;
		case WFR2S: 
			meco_fun    = meco_fun_wfr2s;
			// scaling[2]  = 0.75 + 0.0 * I; // R2*
			break;
		case WF2R2S: 
			meco_fun    = meco_fun_wf2r2s;
			// scaling[1]  = 0.75 + 0.0 * I; // R2*_W
			// scaling[3]  = 0.75 + 0.0 * I; // R2*_F
			break;
		case R2S: 
			meco_fun    = meco_fun_r2s;
			// scaling[1]  = 0.75 + 0.0 * I; // R2*
			break;
	}

	data->scaling = md_alloc(N, scaling_dims, CFL_SIZE);
	md_copy(N, scaling_dims, data->scaling, scaling, CFL_SIZE);
	xfree(scaling);

	return nlop_create(N, y_dims, N, x_dims, CAST_UP(PTR_PASS(data)), meco_fun, meco_der, meco_adj, NULL, NULL, meco_del);
}

