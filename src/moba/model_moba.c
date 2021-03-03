
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"

#include "moba/moba.h"
#include "moba/T1fun.h"
#include "moba/T2fun.h"
#include "moba/IR_SS_fun.h"
#include "moba/T1MOLLI.h"
#include "moba/T1_alpha.h"
#include "moba/T1_alpha_in.h"
#include "moba/blochfun.h"
#include "moba/model_Bloch.h"

#include "model_moba.h"


static void bloch_struct_conversion(const long dims[DIMS], struct modBlochFit* out, const struct sim_conf_s* in)
{
	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, map_dims, dims);

	out->sequence = (int)in->sequence;
	out->rfduration = in->rfduration;
	out->bwtp = in->bwtp;
	out->tr = in->tr;
	out->te = in->te;
	out->averaged_spokes = in->averaged_spokes;
	out->sliceprofile_spins = in->sliceprofile_spins;
	out->num_vfa = in->num_vfa;
	out->fa = in->fa;
	out->runs = in->runs;
	out->inversion_pulse_length = in->inversion_pulse_length;
	out->prep_pulse_length = in->prep_pulse_length;
	out->look_locker_assumptions = in->look_locker_assumptions;

	memcpy(out->scale, in->scale, sizeof(in->scale));
	out->fov_reduction_factor = in->fov_reduction_factor;
	out->rm_no_echo = in->rm_no_echo;
	out->full_ode_sim = (int)in->sim_type;
	out->not_wav_maps = in->not_wav_maps;

	if (NULL != in->input_b1) {
		out->input_b1 = md_alloc(DIMS, map_dims, CFL_SIZE);
		md_copy(DIMS, map_dims, out->input_b1, in->input_b1, CFL_SIZE);
	}

	long sp_dims[DIMS];
	md_set_dims(DIMS, sp_dims, 1);
	sp_dims[READ_DIM] = in->sliceprofile_spins;

	if (NULL != in->input_sliceprofile) {
		out->input_sliceprofile = md_alloc(DIMS, sp_dims, CFL_SIZE);
		md_copy(DIMS, sp_dims, out->input_sliceprofile, in->input_sliceprofile, CFL_SIZE);
	}
	// FIXME: Add later
	// out->input_fa_profile = NULL;
}


struct moba_s moba_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf, const struct moba_conf_s* conf_model, _Bool use_gpu)
{
	// FIXME
	// function to get TI from moba_conf and dims
	//...complex float* TI = NULL;

	// function receiving B1 and outputting alpha
	//...complex float* alpha = NULL;

	// function calculating the slice profile if desired


	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create3(data_dims, mask, psf, conf);
	struct moba_s ret;

	// FIXME: unify them more
	long der_dims[DIMS];
	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|COEFF_FLAG|TIME2_FLAG, der_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TIME_FLAG|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG|TIME_FLAG|TIME2_FLAG, TI_dims, dims);


#if 1

	struct nlop_s* model = NULL;
	struct modBlochFit fitpara = modBlochFit_defaults;

	switch (conf_model->model) {

	case IR:

		model = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case MOLLI:

		//FIXME: Currently MOLLI model supports an acquisition of 5 heart beats
		out_dims[TE_DIM] /= 5;
		TI_dims[TE_DIM] /= 5;

		complex float* TI1 = md_alloc(DIMS, TI_dims, CFL_SIZE);
		complex float* TI2 = md_alloc(DIMS, TI_dims, CFL_SIZE);

		md_copy(DIMS, TI_dims, TI1, conf_model->irflash.input_TI, CFL_SIZE);
		md_copy(DIMS, TI_dims, TI2, conf_model->irflash.input_TI, CFL_SIZE);

        	model = nlop_T1MOLLI_create(DIMS, map_dims, out_dims, TI_dims, TI1, TI2, conf_model->irflash.input_TI_t1relax, use_gpu);

        	md_free(TI1);
		md_free(TI2);
		break;

	case IR_SS:

		model = nlop_IR_SS_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case IR_phy:

		model = nlop_T1_alpha_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case IR_phy_alpha_in:

		model = nlop_T1_alpha_in_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, conf_model->irflash.input_alpha, use_gpu);
		break;

	case T2:

		model = nlop_T2_create(DIMS, map_dims, out_dims, in_dims, TI_dims, conf_model->irflash.input_TI, use_gpu);
		break;

	case MGRE:

		error("Model not yet supported by moba/model_moba.c");
		break;

	case Bloch:

		bloch_struct_conversion(dims, &fitpara, &(conf_model->sim));

		if (5 == fitpara.sequence) {

			// Turn of matching of T2 for IR FLASH
			fitpara.scale[2] = 0.0001;

			// Simulate Look-Locker assumption: Echo(t=TE) == Mz(t=0)
			fitpara.look_locker_assumptions = true;
		}

		model = nlop_Bloch_create(DIMS, der_dims, map_dims, out_dims, in_dims, &fitpara, use_gpu);
		break;
	}

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(model, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(model, 0)->dims);

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(model, 0, b, 0);
	nlop_free(b);

	if (MOLLI == conf_model->model)
		nlinv.nlop = nlop_permute_inputs(c, 4, (const int[4]){ 1, 2, 3, 0 });
	else
		nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });

	nlop_free(c);

#endif
	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;

	if (IR_phy == conf_model->model)
		ret.linop_alpha = T1_get_alpha_trafo(model);

	nlop_free(nlinv.nlop);

	return ret;
}


