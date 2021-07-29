
#include <string.h>

#include "iter/italgos.h"

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
#include "nlops/stack.h"

#include "nn/nn.h"
#include "nn/losses.h"
#include "nn/chain.h"

#include "networks/losses.h"

struct loss_config_s valid_loss_nlinvnet = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 1.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_mse_rss = 1.,
	.weighting_psnr_rss = 1.,
	.weighting_ssim_rss = 1.,

	.weighting_cce = 0.,
	.weighting_weighted_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
	.image_flags = FFT_FLAGS,
};

struct loss_config_s train_loss_nlinvnet = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_mse_rss = 0.,
	.weighting_psnr_rss = 0.,
	.weighting_ssim_rss = 0.,

	.weighting_cce = 0.,
	.weighting_weighted_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
	.image_flags = FFT_FLAGS,
};

struct loss_config_s loss_option = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_mse_rss = 0.,
	.weighting_psnr_rss = 0.,
	.weighting_ssim_rss = 0.,

	.weighting_cce = 0.,
	.weighting_weighted_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
	.image_flags = FFT_FLAGS,
};

struct loss_config_s val_loss_option = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_mse_rss = 0.,
	.weighting_psnr_rss = 0.,
	.weighting_ssim_rss = 0.,

	.weighting_cce = 0.,
	.weighting_weighted_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.label_index = 0,
	.image_flags = FFT_FLAGS,
};

struct loss_config_s loss_image_valid = {

	.weighting_mse_sa = 1.,
	.weighting_mse = 1.,
	.weighting_psnr = 1.,
	.weighting_ssim = 1.,

	.weighting_mse_rss = 0.,
	.weighting_psnr_rss = 0.,
	.weighting_ssim_rss = 0.,

	.weighting_cce = 0.,
	.weighting_weighted_cce = 0.,
	.weighting_accuracy = 0.,

	.weighting_dice0 = 0.,
	.weighting_dice1 = 0.,
	.weighting_dice2 = 0.,

	.weighting_dice_labels = 0.,

	.label_index = 0,
	.image_flags = FFT_FLAGS,
};

struct loss_config_s loss_classification_valid = {

	.weighting_mse_sa = 0.,
	.weighting_mse = 0.,
	.weighting_psnr = 0.,
	.weighting_ssim = 0.,

	.weighting_mse_rss = 0.,
	.weighting_psnr_rss = 0.,
	.weighting_ssim_rss = 0.,

	.weighting_cce = 1.,
	.weighting_weighted_cce = 1.,
	.weighting_accuracy = 1.,

	.weighting_dice0 = 1.,
	.weighting_dice1 = 1.,
	.weighting_dice2 = 1.,

	.weighting_dice_labels = 1.,

	.label_index = 0,
	.image_flags = FFT_FLAGS,
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

static const struct nlop_s* nlop_affine_transform_out_F(const struct nlop_s* nlop, complex float a, complex float b)
{
	assert(1 == nlop_get_nr_out_args(nlop));
	int N = nlop_generic_codomain(nlop, 0)->N;
	const long* dims = nlop_generic_codomain(nlop, 0)->dims;

	if (0 == b)
		return nlop_chain2_FF(nlop, 0, nlop_from_linop_F(linop_scale_create(N, dims, a)), 0);

	complex float* tmp = md_alloc(N, dims, CFL_SIZE);
	md_zfill(N, dims, tmp, b);

	auto result = nlop_zaxpbz_create(N, dims, a, 1);
	result = nlop_set_input_const_F(result, 1, N, dims, true, tmp);

	result = nlop_chain2_FF(nlop, 0, result, 0);

	md_free(tmp);
	return result;
}

static nn_t nlop_loss_to_nn_F(const struct nlop_s* nlop, const char* name, float weighting, bool measure)
{
	if (measure && 1 != weighting)
		error("Scaling other than 0. and 1. is only allowed for losses!");

	auto tmp_loss = nn_from_nlop_F(nlop_chain2_FF(nlop, 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), weighting)), 0));
	tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
	tmp_loss = nn_set_output_name_F(tmp_loss, 0, name);
	return tmp_loss;
}

static nn_t loss_measure_create(const struct loss_config_s* config, unsigned int N, const long dims[N], bool combine, bool measure)
{
	UNUSED(dims);

	nn_t result = NULL;

	if (0 != config->weighting_mse_sa) {

		const struct nlop_s* tmp_loss_nlop = nlop_mse_create(N, dims, ~0ul);
		tmp_loss_nlop = nlop_chain2_FF(nlop_smo_abs_create(N, dims, measure ? 0 : 1.e-12), 0, tmp_loss_nlop, 0);
		tmp_loss_nlop = nlop_chain2_FF(nlop_smo_abs_create(N, dims, measure ? 0 : 1.e-12), 0, tmp_loss_nlop, 0);

		result = add_loss(result, nlop_loss_to_nn_F(tmp_loss_nlop, "mse mag", config->weighting_mse_sa, measure), combine);
	}

	if (0 != config->weighting_mse) {

		result = add_loss(result, nlop_loss_to_nn_F(nlop_mse_create(N, dims, ~0ul), "mse", config->weighting_mse, measure), combine);
	}

	if (0 != config->weighting_psnr) {

		auto nlop = nlop_mpsnr_create(N, dims, ~config->image_flags);

		if (!measure) {

			assert(0); //cannot be used for training
			nlop = nlop_affine_transform_out_F(nlop, -1, 0);
		}

		result = add_loss(result, nlop_loss_to_nn_F(nlop, "mean psnr", config->weighting_psnr, measure), combine);
	}

	if (0 != config->weighting_ssim) {

		assert(5 <= N); //FIXME: should be more general
		assert(0 == (config->image_flags & ~(MD_BIT(4) - 1))); //dim 4 becomes batch / average dim

		long ndims[5];
		md_copy_dims(4, ndims, dims);
		ndims[4] = md_calc_size(N - 4, dims + 4);

		auto nlop = nlop_mssim_create(5, ndims, MD_DIMS(7, 7, 1, 1, 1), config->image_flags);

		if (!measure)
			nlop = nlop_affine_transform_out_F(nlop, -1, 1);

		nlop = nlop_reshape_in_F(nlop, 0, N, dims);
		nlop = nlop_reshape_in_F(nlop, 1, N, dims);

		result = add_loss(result, nlop_loss_to_nn_F(nlop, "mean ssim", config->weighting_ssim, measure), combine);
	}

	if (0 != config->weighting_mse_rss) {

		long loss_dims[N];
		md_select_dims(N, ~COIL_FLAG, loss_dims, dims);

		auto loss = nlop_mse_create(N, loss_dims, ~0ul);
		loss = nlop_chain2_FF(nlop_zrss_create(N, dims, COIL_FLAG, measure ? 0 : 1.e-12), 0, loss, 0);
		loss = nlop_chain2_FF(nlop_zrss_create(N, dims, COIL_FLAG, measure ? 0 : 1.e-12), 0, loss, 0);

		result = add_loss(result, nlop_loss_to_nn_F(loss, "mse rss", config->weighting_mse_rss, measure), combine);
	}

	if (0 != config->weighting_psnr_rss) {

		long loss_dims[N];
		md_select_dims(N, ~COIL_FLAG, loss_dims, dims);

		auto nlop = nlop_mpsnr_create(N, loss_dims, ~config->image_flags);

		if (!measure) {

			assert(0); //cannot be used for training
			nlop = nlop_affine_transform_out_F(nlop, -1, 0);
		}

		nlop = nlop_chain2_FF(nlop_zrss_create(N, dims, COIL_FLAG, measure ? 0 : 1.e-12), 0, nlop, 0);
		nlop = nlop_chain2_FF(nlop_zrss_create(N, dims, COIL_FLAG, measure ? 0 : 1.e-12), 0, nlop, 0);

		result = add_loss(result, nlop_loss_to_nn_F(nlop, "mean psnr rss", config->weighting_psnr_rss, measure), combine);
	}

	if (0 != config->weighting_ssim_rss) {

		assert(5 <= N); //FIXME: should be more general
		assert(0 == (config->image_flags & ~(MD_BIT(4) - 1))); //dim 4 becomes batch / average dim

		long loss_dims[N];
		md_select_dims(N, ~COIL_FLAG, loss_dims, dims);

		long ndims[5];
		md_copy_dims(4, ndims, loss_dims);
		ndims[4] = md_calc_size(N - 4, dims + 4);

		auto nlop = nlop_mssim_create(5, ndims, MD_DIMS(7, 7, 1, 1, 1), config->image_flags);

		if (!measure)
			nlop = nlop_affine_transform_out_F(nlop, -1, 1);

		nlop = nlop_reshape_in_F(nlop, 0, N, loss_dims);
		nlop = nlop_reshape_in_F(nlop, 1, N, loss_dims);

		nlop = nlop_chain2_FF(nlop_zrss_create(N, dims, COIL_FLAG, measure ? 0 : 1.e-12), 0, nlop, 0);
		nlop = nlop_chain2_FF(nlop_zrss_create(N, dims, COIL_FLAG, measure ? 0 : 1.e-12), 0, nlop, 0);

		result = add_loss(result, nlop_loss_to_nn_F(nlop, "mean ssim rss", config->weighting_ssim_rss, measure), combine);
	}

	if (0 != config->weighting_cce) {

		if (0 > config->label_index)
			error("Label index not set!");

		auto nlop = nlop_cce_create(N, dims, ~MD_BIT(config->label_index));

		result = add_loss(result, nlop_loss_to_nn_F(nlop, "cce", config->weighting_cce, measure), combine);
	}

	if (0 != config->weighting_weighted_cce) {

		if (0 > config->label_index)
			error("Label index not set!");

		auto nlop = nlop_weighted_cce_create(N, dims, ~MD_BIT(config->label_index));

		result = add_loss(result, nlop_loss_to_nn_F(nlop, "weighted cce", config->weighting_weighted_cce, measure), combine);
	}

	if (0 != config->weighting_accuracy) {

		if (0 > config->label_index)
			error("Label index not set!");

		if (!measure)
			error("Accuracy cannot be used as training loss!");

		auto nlop = nlop_accuracy_create(N, dims, config->label_index);

		result = add_loss(result, nlop_loss_to_nn_F(nlop, "accuracy", config->weighting_accuracy, measure), combine);
	}

	if (0 != config->weighting_dice0) {

		if (0 > config->label_index)
			error("Label index not set!");

		auto nlop = nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, 0., false);
		if (measure)
			nlop = nlop_affine_transform_out_F(nlop, -1, 1);

		result = add_loss(result, nlop_loss_to_nn_F(nlop, measure ? "dice sim 0" : "dice loss 0", config->weighting_dice0, measure), combine);
	}

	if (0 != config->weighting_dice1) {

		if (0 > config->label_index)
			error("Label index not set!");

		auto nlop = nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, -1., false);
		if (measure)
			nlop = nlop_affine_transform_out_F(nlop, -1, 1);

		result = add_loss(result, nlop_loss_to_nn_F(nlop, measure ? "dice sim 1" : "dice loss 1", config->weighting_dice1, measure), combine);
	}

	if (0 != config->weighting_dice2) {

		if (0 > config->label_index)
			error("Label index not set!");

		auto nlop = nlop_dice_create(N, dims, MD_BIT(config->label_index), 0, -2., false);
		if (measure)
			nlop = nlop_affine_transform_out_F(nlop, -1, 1);

		result = add_loss(result, nlop_loss_to_nn_F(nlop, measure ? "dice sim 2" : "dice loss 2", config->weighting_dice2, measure), combine);
	}


	if (0 != config->weighting_dice_labels) {

		if (0 > config->label_index)
			error("Label index not set!");

		long labels = dims[config->label_index];

		auto dice = nlop_dice_generic_create(N, dims, MD_BIT(config->label_index), MD_BIT(config->label_index), 0., false);
		dice = nlop_reshape_out_F(dice, 0, 1, MD_DIMS(md_calc_size(N, nlop_generic_codomain(dice, 0)->dims)));

		if (measure)
			dice = nlop_affine_transform_out_F(dice, -1, 1);
		else
			error("Dice labels are only supported as measure!");

		dice = nlop_chain2_FF(dice, 0, nlop_from_linop_F(linop_scale_create(1, MD_DIMS(labels), config->weighting_dice_labels)), 0);
		while (labels > 1) {

			dice = nlop_chain2_FF(dice, 0, nlop_destack_create(1, MD_DIMS(labels - 1), MD_DIMS(1), MD_DIMS(labels), 0), 0);
			labels--;
		}
		labels = 0;

		nn_t tmp_loss = nn_from_nlop_F(dice);

		while (0 < nn_get_nr_unnamed_out_args(tmp_loss)) {

			auto name = ptr_printf("dice sim (label %d)", labels++);
			tmp_loss = nn_set_out_type_F(tmp_loss, 0, NULL, OUT_OPTIMIZE);
			tmp_loss = nn_set_output_name_F(tmp_loss, 0, name);
			xfree(name);
		}

		result = add_loss(result, tmp_loss, combine);
	}

	return result;
}

nn_t train_loss_create(const struct loss_config_s* config, unsigned int N, const long dims[N])
{
	return loss_measure_create(config, N, dims, true, false);
}

nn_t val_measure_create(const struct loss_config_s* config, unsigned int N, const long dims[N])
{
	return loss_measure_create(config, N, dims, false, true);
}

/**
 * Returns sum of losses
 *
 * [sum_i (weighting_base)^(M - i) L(in_i, ref) ] / [sum_i (weighting_base)^(M - i)]
 *
 *
 **/
nn_t train_loss_multi_create(const struct loss_config_s* config, unsigned int N, const long dims[N], int M, float weighting_base)
{
	auto result = train_loss_create(config, N, dims);

	if (0 >= weighting_base) {

		for (int i = 1; i < M; i++)
			result = nn_combine_FF(nn_from_nlop_F(nlop_del_out_create(N, dims)), result);

		return result;
	}

	const char* loss_name = nn_get_out_name_from_arg_index(result, 0, true);

	for (int i = 1; i < M; i++) {

		auto nn_tmp = nn_from_nlop_F(nlop_zaxpbz_create(1, MD_DIMS(1), weighting_base  / (1. + weighting_base), 1. / (1. + weighting_base)));

		//result: in: in_0, ..., in_i-1, ref; out: old_loss

		result = nn_chain2_FF(result, 0, loss_name, nn_tmp, 0, NULL); //in: new_loss, in_0, ..., in_i-1, ref; out:  s1 * new_loss + s2 * old_loss
		result = nn_chain2_FF(train_loss_create(config, N, dims), 0, loss_name, result, 0, NULL); //in: in_0, ..., in_i-1, ref, in_i, ref; out:  s1 * new_loss + s2 * old_loss
		result = nn_dup_F(result, i, NULL, i + 2, NULL); //in: in_0, ..., in_i-1, ref, in_i; out:  s1 * new_loss + s2 * old_loss
		result = nn_shift_input_F(result, i + 1, NULL, i, NULL); //in: in_0, ..., in_i-1, in_i, ref; out: s1 * new_loss + s2 * old_loss

		result = nn_set_output_name_F(result, 0, loss_name);
		result = nn_set_out_type_F(result, 0, loss_name, OUT_OPTIMIZE);
	}

	xfree(loss_name);

	return result;
}