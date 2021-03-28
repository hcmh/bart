
#include "misc/misc.h"
#include "misc/opts.h"
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

struct loss_config_s loss_option = {

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

bool loss_option_changed(struct loss_config_s* loss_option)
{
	if (0 != loss_option->weighting_mse_sa)
		return true;
	if (0 != loss_option->weighting_mse)
		return true;
	if (0 != loss_option->weighting_psnr)
		return true;
	if (0 != loss_option->weighting_ssim)
		return true;

	if (0 != loss_option->weighting_cce)
		return true;
	if (0 != loss_option->weighting_accuracy)
		return true;

	if (0 != loss_option->weighting_dice0)
		return true;
	if (0 != loss_option->weighting_dice1)
		return true;
	if (0 != loss_option->weighting_dice2)
		return true;

	return false;
}

struct opt_s loss_opts[] = {

	OPTL_FLOAT(0, "mse", &(loss_option.weighting_mse), "weighting", "weighting for mean squared error"),
	OPTL_FLOAT(0, "mse_sa", &(loss_option.weighting_mse_sa), "weighting", "weighting for smoothed mean squared error of magnitude"),
	//OPTL_FLOAT(0, "psnr", &(loss_option.weighting_psnr), "weighting", "weighting for peak signal to noise ratio (no training)"),
	//OPTL_FLOAT(0, "ssim", &(loss_option.weighting_ssim), "weighting", "weighting for structural similarity index measure (no training)"),

	OPTL_FLOAT(0, "cce", &(loss_option.weighting_cce), "weighting", "weighting for categorical cross entropy"),
	//OPTL_FLOAT(0, "acc", &(loss_option.weighting_accuracy), "weighting", "weighting for accuracy (no training)"),

	OPTL_FLOAT(0, "dice0", &(loss_option.weighting_dice0), "weighting", "weighting for unbalanced dice loss"),
	OPTL_FLOAT(0, "dice1", &(loss_option.weighting_dice1), "weighting", "weighting for dice loss weighted with inverse frequency of label"),
	OPTL_FLOAT(0, "dice2", &(loss_option.weighting_dice2), "weighting", "weighting for dice loss weighted with inverse square frequency of label"),

	//OPTL_FLOAT(0, "dicel", &(loss_option.weighting_dice2), "weighting", "weighting for per lable dice loss"),

	OPTL_UINT(0, "label_dim", &(loss_option.label_index), "index", "label dimension"),
};

const int N_loss_opts = ARRAY_SIZE(loss_opts);


struct loss_config_s val_loss_option = {

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


struct opt_s val_loss_opts[] = {

	OPTL_FLOAT(0, "mse", &(val_loss_option.weighting_mse), "weighting", "weighting for mean squared error"),
	OPTL_FLOAT(0, "mse_sa", &(val_loss_option.weighting_mse_sa), "weighting", "weighting for smoothed mean squared error of magnitude"),
	OPTL_FLOAT(0, "psnr", &(val_loss_option.weighting_psnr), "weighting", "weighting for peak signal to noise ratio (no training)"),
	OPTL_FLOAT(0, "ssim", &(val_loss_option.weighting_ssim), "weighting", "weighting for structural similarity index measure (no training)"),

	OPTL_FLOAT(0, "cce", &(val_loss_option.weighting_cce), "weighting", "weighting for categorical cross entropy"),
	OPTL_FLOAT(0, "acc", &(val_loss_option.weighting_accuracy), "weighting", "weighting for accuracy (no training)"),

	OPTL_FLOAT(0, "dice0", &(val_loss_option.weighting_dice0), "weighting", "weighting for unbalanced dice loss"),
	OPTL_FLOAT(0, "dice1", &(val_loss_option.weighting_dice1), "weighting", "weighting for dice loss weighted with inverse frequency of label"),
	OPTL_FLOAT(0, "dice2", &(val_loss_option.weighting_dice2), "weighting", "weighting for dice loss weighted with inverse square frequency of label"),

	OPTL_FLOAT(0, "dicel", &(val_loss_option.weighting_dice2), "weighting", "weighting for per lable dice loss"),

	OPTL_UINT(0, "label_dim", &(val_loss_option.label_index), "index", "label dimension"),
};

const int N_val_loss_opts = ARRAY_SIZE(val_loss_opts);


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


static nn_t add_loss(nn_t loss, nn_t new_loss, bool combine) {

	auto result = loss;

	if (NULL == result) {

		result = new_loss;
	} else {

		result = nn_combine_FF(result, new_loss);
		result = nn_dup_F(result, 0, NULL, 2, NULL);
		result = nn_dup_F(result, 1, NULL, 2, NULL);

	}

	while (combine) {

		combine = false;
		int i_loss = -1;
		int j_loss = -1;

		int OO = nn_get_nr_out_args(result);
		enum OUT_TYPE out_types[OO];
		nn_get_out_types(result, OO, out_types);

		for (int i = 0; (-1 == j_loss) && (i < OO); i++)
			if (OUT_OPTIMIZE == out_types[i]) {

				if (-1 == i_loss)
					i_loss = i;
				else
				 	j_loss = i;
			}

		if (-1 != j_loss) {

			int i_index = nn_get_out_index_from_arg_index(result, i_loss);
			int j_index = nn_get_out_index_from_arg_index(result, j_loss);
			const char* i_name = nn_get_out_name_from_arg_index(result, i_loss, true);
			const char* j_name = nn_get_out_name_from_arg_index(result, j_loss, true);

			auto sum = nn_from_nlop_F(nlop_zaxpbz_create(1, MD_DIMS(1), 1, 1));
			result = nn_combine_FF(sum, result);
			result = nn_link_F(result, j_index, j_name, 0, NULL);
			result = nn_link_F(result, i_index, i_name, 0, NULL);

			const char* nname = ptr_printf("%s + %s", (NULL == i_name) ? "na" : i_name, (NULL == j_name) ? "na" : j_name);

			result = nn_set_out_type_F(result, 0, NULL, OUT_OPTIMIZE);
			result = nn_set_output_name_F(result, 0, nname);

			xfree(nname);

			if (NULL != i_name)
				xfree(i_name);
			if (NULL != j_name)
				xfree(j_name);

			combine = true;
		}
	}

	return result;
}

nn_t loss_create(const struct loss_config_s* config, unsigned int N, const long dims[N], bool combine)
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

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_mse) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_mse_create(N, dims, ~0ul), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_mse)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "mse");

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_psnr) {

		assert(5 == N); //FIXME: should be more general
		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_mpsnr_create(N, dims, MD_BIT(4)), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_psnr)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "mpsnr");

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_ssim) {

		assert(5 == N); //FIXME: should be more general
		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_mssim_create(N, dims, MD_DIMS(7, 7, 1, 1, 1), FFT_FLAGS), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_ssim)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "mssim");

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_cce) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_cce_create(N, dims, ~MD_BIT(config->label_index)), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_cce)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "cce");

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_accuracy) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_accuracy_create(N, dims, config->label_index), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_cce)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "accuracy");

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_dice0) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, 0., false), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_dice0)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "dice0");

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_dice1) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, -1., false), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_dice1)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "dice1");

		result = add_loss(result, tmp_loss, combine);
	}

	if (0 != config->weighting_dice2) {

		nn_t tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, -2., false), 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), config->weighting_dice2)), 0));
		tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
		tmp_loss = nn_set_output_name_F(tmp_loss, 0, "dice2");

		result = add_loss(result, tmp_loss, combine);
	}

	return result;
}