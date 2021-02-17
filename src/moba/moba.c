/* Copyright 2019-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/
 
#include <stdbool.h>
#include <stddef.h>

#include "moba.h"


const struct sim_conf_s sim_conf_s_defaults = {

	.sequence = 1, /*inv. bSSFP*/
	.rfduration = 0.0009,
	.bwtp = 4,
	.tr = 0.0045,
	.te = 0.00225,
	.averaged_spokes = 1,
	.sliceprofile_spins = 1,
	.num_vfa = 1,
	.fa = 45.,
	.runs = 1,
	.inversion_pulse_length = 0.01,
	.prep_pulse_length = 0.00225,

	.scale = {1., 1., 1., 1.},
	.fov_reduction_factor = 1.,
	.rm_no_echo = 0.,
	.sim_type = OBS,
	.not_wav_maps = 0,

	.input_b1 = NULL,
	.input_sliceprofile = NULL,
	.input_fa_profile = NULL,
};

const struct irflash_conf_s irflash_conf_s_defaults = {

	.input_TI = NULL,

	.input_TI_t1relax = NULL,

	.input_alpha = NULL,
};

struct opt_conf_s opt_conf_s_defaults = {

	.iter = 8,
	.opt_reg = 1.,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.step = 0.9,
	.lower_bound = 0.,
	.tolerance = 0.01,
	.damping = 0.9,
	.inner_iter = 250,
	.noncartesian = false,
	.sms = false,
        .k_filter = false,
	.auto_norm_off = false,
	.algo = 3,
	.rho = 0.01,
	.stack_frames = false,
};

struct moba_conf moba_defaults = {

	.iter = 8,
	.opt_reg = 1.,
	.alpha = 1.,
	.alpha_min = 0.,
	.redu = 2.,
	.step = 0.9,
	.lower_bound = 0.,
	.tolerance = 0.01,
	.damping = 0.9,
	.inner_iter = 250,
	.noncartesian = false,
	.sms = false,
        .k_filter = false,
	.MOLLI = false,
	.IR_SS = false,
	.IR_phy = 0.,
	.auto_norm_off = false,
	.algo = 3,
	.rho = 0.01,
	.stack_frames = false,
	.input_alpha = NULL,
};

