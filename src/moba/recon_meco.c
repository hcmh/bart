
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "num/gpuops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/rand.h"

#include "iter/italgos.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/vec.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "nlops/nlop.h"

#include "noir/model.h"

#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"

#include "meco.h"
#include "model_meco.h"
#include "recon_meco.h"


struct mecoinv_s {

	INTERFACE(iter_op_data);

	unsigned int model;
	unsigned int regu;

	const struct nlop_s* nlop;
	const struct iter3_irgnm_conf* conf;

	long x_size;
	long y_size;

	const long* x_dims;

	float alpha;

	bool first_iter;
	int outer_iter;

	const struct operator_p_s* prox1;
	const struct operator_p_s* prox2;
};

DEF_TYPEID(mecoinv_s);



// TODO: the src and dst must be the same pointer, otherwise a copy from src to dst must be done in the begining of the function
static void nonneg_constraint(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(mecoinv_s, _data);

	long map_size = data->x_dims[0] * data->x_dims[1];

	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, map_dims, data->x_dims);

	long nmaps = data->x_dims[COEFF_DIM];
	long dista = 0;


	float lower_bound = 0.0;

	if ((data->model == WFR2S) || (data->model == R2S)) {

		dista = (nmaps - 2) * map_size;
		md_zsmax(DIMS, map_dims, (_Complex float*)dst + dista, (const _Complex float*)src + dista, lower_bound);

	} else
	if ( data->model == WF2R2S ) {

		dista = 3 * map_size;
		md_zsmax(DIMS, map_dims, (_Complex float*)dst + dista, (const _Complex float*)src + dista, lower_bound);

		dista = 1 * map_size;
		md_zsmax(DIMS, map_dims, (_Complex float*)dst + dista, (const _Complex float*)src + dista, lower_bound);

	}

}


// TODO: the src and dst must be the same pointer, otherwise a copy from src to dst must be done in the begining of the function
static void combined_prox(iter_op_data* _data, float rho, float* dst, const float* src)
{
	auto data = CAST_DOWN(mecoinv_s, _data);

	if (data->first_iter) {

		data->first_iter = false;

	} else {

		nonneg_constraint(_data, dst, src);
	}

#if 1
	operator_p_apply_unchecked(data->prox2, rho, (_Complex float*)dst, (const _Complex float*)dst);
#else
	operator_p_apply_unchecked(data->prox1, rho, (_Complex float*)dst, (const _Complex float*)dst);
#endif

	nonneg_constraint(_data, dst, dst);
}



static void normal_equ(iter_op_data* _data, float* dst, const float* src)
{
	auto data = CAST_DOWN(mecoinv_s, _data);

	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->y_size), FL_SIZE, src);

	linop_forward_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);
	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)dst, (const complex float*)tmp);

	md_free(tmp);

#if 0
	long map_size = data->x_dims[0] * data->x_dims[1];
	long maps = data->x_dims[COEFF_DIM];
	long sens = data->x_dims[COIL_DIM];

	md_axpy(1, MD_DIMS(data->x_size * sens / (maps + sens)),
						 dst + map_size * 2 * maps,
						 data->alpha,
						 src + map_size * 2 * maps);
#else
	md_axpy(1, MD_DIMS(data->x_size), dst, data->alpha, src);
#endif
}



static void inverse_fista(iter_op_data* _data, float alpha, float* dst, const float* src)
{
	auto data = CAST_DOWN(mecoinv_s, _data);

	data->alpha = alpha;	// update alpha for normal operator

    
	void* x = md_alloc_sameplace(1, MD_DIMS(data->x_size), FL_SIZE, src);
	md_gaussian_rand(1, MD_DIMS(data->x_size / 2), x);
	double maxeigen = power(20, data->x_size, select_vecops(src), (struct iter_op_s){ normal_equ, CAST_UP(data) }, x);
	md_free(x);

	double step = data->conf->step / maxeigen;

	if (WAV == data->regu)
		wavthresh_rand_state_set(data->prox1, 1);
    
	int maxiter = MIN(data->conf->cgiter, 10 * powf(2, data->outer_iter));
    
	float* tmp = md_alloc_sameplace(1, MD_DIMS(data->y_size), FL_SIZE, src);

	linop_adjoint_unchecked(nlop_get_derivative(data->nlop, 0, 0), (complex float*)tmp, (const complex float*)src);

	float eps = md_norm(1, MD_DIMS(data->x_size), tmp);

	data->first_iter = true;

	NESTED(void, continuation, (struct ist_data* itrdata))
	{
		itrdata->scale = data->alpha;
	};

	debug_printf(DP_DEBUG3, "> FISTA parameters:\n");
	debug_printf(DP_DEBUG3, "  alpha %.6f; epsilon %.6f; tau %.6f\n", alpha, data->conf->cgtol * alpha * eps, step);

	fista(maxiter, data->conf->cgtol * alpha * eps, step,
		data->x_size,
		select_vecops(src),
		continuation,
		(struct iter_op_s){ normal_equ, CAST_UP(data) },
		(struct iter_op_p_s){ combined_prox, CAST_UP(data) },
		dst, tmp, NULL);

	nonneg_constraint(CAST_UP(data), dst, dst);

	md_free(tmp);

	data->outer_iter++;
}



static const struct operator_p_s* create_wav_prox(const long img_dims[DIMS], unsigned int jt_flag)
{
	bool randshift = true;
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	unsigned int wflags = 0;

	for (unsigned int i = 0; i < DIMS; i++) {

		if ((1 < img_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {

			wflags = MD_SET(wflags, i);
			minsize[i] = MIN(img_dims[i], DIMS);
		}
	}

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, jt_flag, minsize, 1., randshift);
}

static const struct operator_p_s* create_llr_prox(const long img_dims[DIMS], unsigned int jt_flag)
{
	bool randshift = true;
	long blk_dims[MAX_LEV][DIMS];
	int blk_size = 8;

	int levels = llr_blkdims(blk_dims, ~jt_flag, img_dims, blk_size);
	UNUSED(levels);

	debug_printf(DP_DEBUG2, "  > blk_dims: ");
	debug_print_dims(DP_DEBUG2, DIMS, blk_dims[0]);

	return lrthresh_create(img_dims, randshift, ~jt_flag, (const long (*)[])blk_dims, 1., false, false, false);
}



struct mecoinv2_s {

	INTERFACE(operator_data_t);

	struct mecoinv_s data;
};

DEF_TYPEID(mecoinv2_s);


static void mecoinv_apply(const operator_data_t* _data, float alpha, complex float* dst, const complex float* src)
{
	const auto data = &CAST_DOWN(mecoinv2_s, _data)->data;
	inverse_fista(CAST_UP(data), alpha, (float*)dst, (const float*)src);
}

static void mecoinv_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(mecoinv2_s, _data);

	operator_p_free(data->data.prox1);
	operator_p_free(data->data.prox2);

	nlop_free(data->data.nlop);

	xfree(data->data.x_dims);
	xfree(data);
}

static const struct operator_p_s* mecoinv_p_create(const struct iter3_irgnm_conf* conf, unsigned int sel_model, unsigned int sel_regu, const long dims[DIMS], struct nlop_s* nlop)
{
	PTR_ALLOC(struct mecoinv2_s, data);
	SET_TYPEID(mecoinv2_s, data);
	SET_TYPEID(mecoinv_s, &data->data);

	auto cd = nlop_codomain(nlop);
	auto dm = nlop_domain(nlop);

	int M = 2 * md_calc_size(cd->N, cd->dims);
	int N = 2 * md_calc_size(dm->N, dm->dims);

	long* ndims = *TYPE_ALLOC(long[DIMS]);
	md_copy_dims(DIMS, ndims, dims);

	long jt_dim = (sel_model == PI) ? (TE_DIM) : (COEFF_DIM);
	unsigned int jt_flag = (sel_model == PI) ? (TE_FLAG) : (COEFF_FLAG);

	long maps_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, maps_dims, dims);
	maps_dims[jt_dim] = (sel_model == PI) ? (maps_dims[jt_dim]) : (maps_dims[jt_dim] - 1);
	
	debug_printf(DP_INFO, "  _ jt_dims: \t");
	debug_print_dims(DP_INFO, DIMS, maps_dims);

	auto prox1 = ((WAV==sel_regu) ? create_wav_prox : create_llr_prox)(maps_dims, jt_flag);

	auto prox2 = op_p_auto_normalize(prox1, ~jt_flag);

	struct mecoinv_s idata = {

		{ &TYPEID(mecoinv_s) }, sel_model, sel_regu, nlop_clone(nlop), conf,
		N, M, ndims, 1.0, true, 0, prox1, prox2
	};

	data->data = idata;

	return operator_p_create(dm->N, dm->dims, cd->N, cd->dims, CAST_UP(PTR_PASS(data)), mecoinv_apply, mecoinv_del);
}



void meco_recon(const struct noir_conf_s* conf, unsigned int sel_model, unsigned int sel_regu, bool out_origin_maps, const long maps_dims[DIMS], const long sens_dims[DIMS], complex float* x, complex float* xref, const complex float* pattern, const complex float* mask, const complex float* TE, const long ksp_dims[DIMS], const complex float* ksp)
{
	long meco_dims[DIMS];
	long map_dims[DIMS];

	bool use_gpu = false;

#ifdef USE_CUDA
	use_gpu = cuda_ondevice(ksp) ? true : false;
#endif
	

	unsigned int fft_flags = FFT_FLAGS|SLICE_FLAG;
	md_select_dims(DIMS, fft_flags|TE_FLAG, meco_dims, ksp_dims);
	md_select_dims(DIMS, fft_flags, map_dims, ksp_dims);
	
	long maps_size = md_calc_size(DIMS, maps_dims);
	long sens_size = md_calc_size(DIMS, sens_dims);
	long x_size = maps_size + sens_size;
	
	long y_size = md_calc_size(DIMS, ksp_dims);

	// x = (maps; coils)
	// variable which is optimized by the IRGNM
	complex float* x_akt = md_alloc_sameplace(1, MD_DIMS(x_size), CFL_SIZE, ksp);
	md_copy(1, MD_DIMS(x_size), x_akt, x, CFL_SIZE);

	complex float* xref_akt = md_alloc_sameplace(1, MD_DIMS(x_size), CFL_SIZE, ksp);
	md_copy(1, MD_DIMS(x_size), xref_akt, xref, CFL_SIZE);

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = conf->rvc;
	mconf.noncart = conf->noncart;
	mconf.fft_flags = fft_flags;
	mconf.a = conf->a;
	mconf.b = conf->b;
	mconf.cnstcoil_flags = TE_FLAG;

	double start_time = timestamp();
	struct meco_s nl = meco_create(ksp_dims, meco_dims, maps_dims, mask, TE, pattern, sel_model, use_gpu, &mconf);
	double nlsecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "  _ nl of meco Create Time: %.2f s\n", nlsecs);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.alpha_min = conf->alpha_min;
	irgnm_conf.redu = conf->redu;
	irgnm_conf.cgiter = 260;
	irgnm_conf.cgtol = (sel_model == PI) ? 0.1 : 0.01; // 1./3.;
	irgnm_conf.nlinv_legacy = (sel_model == PI) ? true : false;
	irgnm_conf.step = 1.; // 0.9; // 0.475;


	long x_dims[DIMS];
	md_merge_dims(DIMS, x_dims, maps_dims, sens_dims);
	debug_printf(DP_DEBUG2, "  _ x_dims: \t");
	debug_print_dims(DP_DEBUG2, DIMS, x_dims);


	long NMAPS = maps_dims[COEFF_DIM];
	long map_size = md_calc_size(DIMS, map_dims);
	long map_pos;


	const struct operator_p_s* inv_op = NULL;

	// irgnm reconstruction 
	if ( sel_regu == TIKHONOV ) {

		iter4_irgnm(CAST_UP(&irgnm_conf), 
			nl.nlop, 
			x_size * 2, (float*)x_akt, (float*)xref_akt, 
			y_size * 2, (const float*)ksp, 
			NULL, (struct iter_op_s){ NULL, NULL });

	} else {

		inv_op = mecoinv_p_create(&irgnm_conf, sel_model, sel_regu, x_dims, nl.nlop);

		iter4_irgnm2(CAST_UP(&irgnm_conf), 
			nl.nlop, 
			x_size * 2, (float*)x_akt, (float*)xref_akt, 
			y_size * 2, (const float*)ksp, 
			inv_op, (struct iter_op_s){ NULL, NULL });
	}


	md_copy(1, MD_DIMS(x_size), xref_akt, x_akt, CFL_SIZE);

	if ( !out_origin_maps ) {

		// fB0 (Hz)
		if ( sel_model != PI ) {
			map_pos = map_size * (NMAPS-1);
			meco_forw_fB0(nl.linop_fB0, xref_akt + map_pos, xref_akt + map_pos);
			md_zsmul(1, MD_DIMS(map_size), xref_akt + map_pos, xref_akt + map_pos, 1000.);
		} 

		// R2s (Hz)
		if ((sel_model == WFR2S) || (sel_model == R2S)) {
			map_pos = map_size * (NMAPS-2);
			md_zsmul(1, MD_DIMS(map_size), xref_akt + map_pos, xref_akt + map_pos, 1000.);

		} else
		if ( sel_model == WF2R2S ) {
			map_pos = map_size * 1;
			md_zsmul(1, MD_DIMS(map_size), xref_akt + map_pos, xref_akt + map_pos, 1000.);

			map_pos = map_size * 3;
			md_zsmul(1, MD_DIMS(map_size), xref_akt + map_pos, xref_akt + map_pos, 1000.);
		}

		noir_forw_coils(nl.linop, xref_akt + maps_size, xref_akt + maps_size);
	}
	

	md_copy(1, MD_DIMS(x_size), xref, xref_akt, CFL_SIZE);

	md_copy(1, MD_DIMS(x_size), x, x_akt, CFL_SIZE);



	md_free(x_akt);
	md_free(xref_akt);

	nlop_free(nl.nlop);

	if (sel_regu != TIKHONOV)
		operator_p_free(inv_op);

	// linop_free(nl.linop);
	// linop_free(nl.linop_fB0);
}