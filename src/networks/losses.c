
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/someops.h"

#include "nn/nn.h"
#include "nn/losses.h"
#include "nn/chain.h"

#include "networks/losses.h"

struct loss_config_s loss_empty = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
};

struct loss_config_s loss_mse = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 1.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
};

struct loss_config_s loss_mse_sa = {

	.weighting_mse_sa = 1.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
};

struct loss_config_s loss_image_valid = {

	.weighting_mse_sa = 1.,
	.weighting_mse = 1.,
	.weighting_psnr = 1.,
	.weighting_ssim = 1.,

	.weighting_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
};

struct loss_config_s loss_classification = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_cce = 1.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
};


struct loss_config_s loss_classification_valid = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_cce = 1.,
	.weighting_accuracy = 1.,

	.weighting_dice0 = 1.,
	.weighting_dice1 = 1.,
	.weighting_dice2 = 1.,

	.label_index = 0,
};


nn_t loss_create(const struct loss_config_s* config, unsigned int N, const long dims[N])
{
	UNUSED(dims);

	nn_t result = NULL;

	if (0 != config->weighting_mse_sa) {

		const struct nlop_s* tmp_loss_nlop = nlop_mse_create(N, dims, ~0ul);
		tmp_loss_nlop = nlop_chain2_FF(nlop_smo_abs_create(N, dims, 1.e-12), 0, tmp_loss_nlop, 0);
		tmp_loss_nlop = nlop_chain2_FF(nlop_smo_abs_create(N, dims, 1.e-12), 0, tmp_loss_nlop, 0);

		auto tmp_loss = nn_from_nlop_F(nlop_chain2_FF(tmp_loss_nlop, 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_mse_sa)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "mse_sa");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_mse) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_mse_create(N, dims, ~0ul), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_mse)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "mse");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_psnr) {

		assert(5 == N); //FIXME: should be more general
		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_mpsnr_create(N, dims, MD_BIT(4)), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_psnr)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "mpsnr");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_ssim) {

		assert(5 == N); //FIXME: should be more general
		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_mssim_create(N, dims, MD_DIMS(7, 7, 1, 1, 1), FFT_FLAGS), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_ssim)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "mssim");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_cce) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_cce_create(N, dims, ~MD_BIT(config->label_index)), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_cce)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "cce");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_accuracy) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_accuracy_create(N, dims, config->label_index), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_cce)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "accuracy");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_dice0) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, 0., false), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_dice0)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "dice0");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_dice1) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, -1., false), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_dice1)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "dice1");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}

	if (0 != config->weighting_dice2) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, -2., false), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_dice2)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "dice2");

		if (NULL == result) {

			result = tmp_loss;
		} else {

			result = nn_combine_FF(result, tmp_loss);
			result = nn_dup_F(result, 0, NULL, 2, NULL);
			result = nn_dup_F(result, 1, NULL, 2, NULL);
		}
	}



	return result;
}