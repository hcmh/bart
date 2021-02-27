/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "num/iovec.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/iter6.h"
#include "iter/italgos.h"
#include "iter/monitor_iter6.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/conv.h"
#include "nlops/someops.h"
#include "nlops/stack.h"

#include "nn/activation.h"
#include "nn/activation_nn.h"
#include "nn/chain.h"
#include "nn/const.h"
#include "nn/layers.h"
#include "nn/layers_nn.h"
#include "nn/nn.h"
#include "nn/weights.h"
#include "nn/losses.h"
#include "nn/losses_nn.h"
#include "nn/init.h"

#include "nn_segm.h"

#include "networks/unet.h"

const struct segm_s segm_default = {

	.Nb = 5,
	.imgx = 128,
	.imgy = 128,
	.classes = 4,

	.in_val = NULL,
	.out_val = NULL,
	.val_loss = NULL,
};

/**
 * Hot encode batch
 * (largest value as index for prediction)
 *
 * @param N_batch batch size
 * @param prediction[(IMG_DIM * IMG_DIM * N_batch)] prediction of single pixels for each image in batch
 * @param N_hotenc number of possible predictions
 * @param label indicator if cfl file should be created for label or prediction
 */
static void hotenc_to_index(int N_batch, long* prediction, const complex float* in, bool cfl, struct segm_s* segm)
{
	long dims[] = { segm->classes, (segm->imgx * segm->imgy * N_batch) };
	long strs[2];

	md_calc_strides(2, strs, dims, CFL_SIZE);

	for (int i_batch = 0; i_batch < (segm->imgx * segm->imgy * N_batch); i_batch++) {

		prediction[i_batch] = 0;

		for (int mask_class = 1; mask_class < segm->classes; mask_class++){

			long pos[] = {mask_class, i_batch};
			long pos_max[] = {prediction[i_batch], i_batch};

			if ((float)MD_ACCESS(2, strs, pos, in) > (float)MD_ACCESS(2, strs, pos_max, in))
				prediction[i_batch] = mask_class;
		}
	}

	if (cfl) {

		complex float* zsegm = xmalloc(sizeof(complex float) * segm->imgx * segm->imgy * N_batch);
		for (int i_batch = 0; i_batch < (segm->imgx * segm->imgy * N_batch); i_batch++)
			zsegm[i_batch] = prediction[i_batch];

		long img_dims[] = {segm->imgx, segm->imgy, N_batch};

		dump_cfl("segmentation_label", 3, img_dims, zsegm);

		xfree(zsegm);
	}
}

static nn_t get_nn_segm_new(int N_batch, struct segm_s* segm, enum NETWORK_STATUS status)
{
	long idims[] = { 1, segm->imgx, segm->imgy, 1, N_batch }; // channel, x, y, z, batch
	long outdims[] = { segm->classes, segm->imgx, segm->imgy, 1, N_batch }; // channel, x, y, z, batch

	return network_unet_create(CAST_UP(&network_unet_default_segm), 5, outdims, 5, idims, status);

	unsigned int N = 5;
	long indims[] = { 1, segm->imgx, segm->imgy, 1, N_batch }; // channel, x, y, z, batch
	long expdims[] = { 2, segm->imgx, segm->imgy, 1, N_batch }; // expand dimensions, channel, x, y, z, batch

	nn_t network = nn_from_nlop_F(nlop_from_linop_F(linop_expand_create(N, expdims, indims)));
	//nn_t network = nn_from_nlop_F(nlop_from_linop(linop_identity_create(N, indims)));

	UNUSED(status);
	long kernel_size[] = { 3, 3, 1 };
	long pool_size[] = { 2, 2, 1 };

	bool conv = true;

	network = nn_append_convcorr_layer(network, 0, NULL, "conv_", 32, kernel_size, conv, PAD_SAME, true, NULL, NULL, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, MD_BIT(0));
	network = nn_append_convcorr_layer(network, 0, NULL, "conv_", 32, kernel_size, conv, PAD_SAME, true, NULL, NULL, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, MD_BIT(0));

	nn_t conv_01 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(network, 0, NULL)->dims))); // create avg and upconv network
	conv_01 = nn_append_maxpool_layer(conv_01, 0, NULL, pool_size, PAD_VALID, true); // 64
	conv_01 = nn_append_convcorr_layer(conv_01, 0, NULL, "conv1_", 64, kernel_size, conv, PAD_SAME, true, NULL, NULL, NULL); // cv_c1; 64

	nn_t conv_02 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_01, 0, NULL)->dims)));
	conv_02 = nn_append_maxpool_layer(conv_02, 0, NULL, pool_size, PAD_VALID, true); // dims: 32
	conv_02 = nn_append_convcorr_layer(conv_02, 0, NULL, "conv2_", 64, kernel_size, conv, PAD_SAME, true, NULL, NULL, NULL); // cv_c2; dims: 32

	nn_t conv_03 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_02, 0, NULL)->dims)));
	conv_03 = nn_append_maxpool_layer(conv_03, 0, NULL, pool_size, PAD_VALID, true); // dims: 16
	conv_03 = nn_append_convcorr_layer(conv_03, 0, NULL, "conv3_", 64, kernel_size, conv, PAD_SAME, true, NULL, NULL, NULL); // cv_c3; dims: 16

	nn_t conv_04 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_03, 0, NULL)->dims)));
	conv_04 = nn_append_maxpool_layer(conv_04, 0, NULL, pool_size, PAD_VALID, true); // dims: 8
	conv_04 = nn_append_convcorr_layer(conv_04, 0, NULL, "conv4_", 64, kernel_size, conv, PAD_SAME, true, NULL, NULL, NULL); // cv_c4; dims: 8

	nn_t conv_05 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_04, 0, NULL)->dims)));
	conv_05 = nn_append_maxpool_layer(conv_05, 0, NULL, pool_size, PAD_VALID, true);// dims: 4
	conv_05 = nn_append_convcorr_layer(conv_05, 0, NULL, "conv5_", 128, kernel_size, conv, PAD_SAME, true, NULL, NULL, NULL); // cv_c5; dims: 4
	conv_05 = nn_append_transposed_convcorr_layer(conv_05, 0, NULL, "transp5_", 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL, NULL); // tr_c5; dims:4
	conv_05 = nn_append_upsampl_layer(conv_05, 0, NULL, pool_size, true); // 8

	nn_t copy_05 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_04, 0, NULL)->dims)));
	nn_t combined_05 = nn_combine_FF(conv_05, copy_05); // in: id_c5, cv_c5, tr_c5, cy_05; out: out_c5, out_cy5
	combined_05 = nn_dup_F(combined_05, 0, NULL, 1, NULL); // in: id_c5, cv_c5, tr_c5; out: out_c5, out_cy5
	combined_05 = nn_combine_FF(nn_from_nlop_F(nlop_zaxpbz_create(5, nn_generic_codomain(conv_04, 0, NULL)->dims, 1.0, 1.0)), combined_05);
	combined_05 = nn_link_F(combined_05, 1, NULL, 0, NULL); // in: out_cy5, id_c5, cv_c5, tr_c5; out: out_c5+out_cy5, out_cy5
	combined_05 = nn_link_F(combined_05, 1, NULL, 0, NULL); // in: id_c5, cv_c5, tr_c5; out: out_c5+out_cy5
	conv_04 = nn_chain2_swap_FF(conv_04, 0, NULL, combined_05, 0, NULL); // in: id_c4, cv_c4, cv_c5, tr_c5; out: out_c5+out_cy5

	conv_04 = nn_append_transposed_convcorr_layer(conv_04, 0, NULL, "transp4_", 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL, NULL); // tr_c4; dims:8
	conv_04 = nn_append_upsampl_layer(conv_04, 0, NULL, pool_size, true); // 16

	nn_t copy_04 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_03, 0, NULL)->dims)));

	nn_t combined_04 = nn_combine_FF(conv_04, copy_04); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4, cy_4; out: out_c4, out_cy4
	combined_04 = nn_dup_F(combined_04, 0, NULL, 1, NULL); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4, out_cy4
	combined_04 = nn_combine_FF(nn_from_nlop_F(nlop_zaxpbz_create(5, nn_generic_codomain(conv_03, 0, NULL)->dims, 1.0, 1.0)), combined_04);
	combined_04 = nn_link_F(combined_04, 1, NULL, 0, NULL); // in: out_c4, id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4+out_cy4, out_cy4
	combined_04 = nn_link_F(combined_04, 1, NULL, 0, NULL); // in: id_c4, C4; out: out_c4+out_cy4
	conv_03 = nn_chain2_swap_FF(conv_03, 0, NULL, combined_04, 0, NULL); // in: id_c3, cv_c3, C4; out: out_c4+out_cy4

	conv_03 = nn_append_transposed_convcorr_layer(conv_03, 0, NULL, "transp3_", 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL, NULL); // tr_c3; dims:8
	conv_03 = nn_append_upsampl_layer(conv_03, 0, NULL, pool_size, true); // 32

	nn_t copy_03 = nn_from_nlop_F(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_02, 0, NULL)->dims)));

	nn_t combined_03 = nn_combine_FF(conv_03, copy_03); // in: id_c3, cv_c3, C4, tr_c3, cy_03; out: out_c3, out_cy3
	combined_03 = nn_dup_F(combined_03, 0, NULL, 1, NULL); // in: id_c3, cv_c3, C4, tr_c3; out: out1, out2
	combined_03 = nn_combine_FF(nn_from_nlop(nlop_zaxpbz_create(5, nn_generic_codomain(conv_02, 0, NULL)->dims, 1.0, 1.0)), combined_03);
	combined_03 = nn_link_F(combined_03, 1, NULL, 0, NULL); // in: out_c3, id_c3, cv_c3, C4, tr_c3; out: out_c3+out_cy3, out_cy3
	combined_03 = nn_link_F(combined_03, 1, NULL, 0, NULL); // in: id_c3, C3; out: out_c3+out_cy3
	conv_02 = nn_chain2_swap_FF(conv_02, 0, NULL, combined_03, 0, NULL); // in: id_c2, cv_c2, C3; out: out_c3+out_cy3

	conv_02 = nn_append_transposed_convcorr_layer(conv_02, 0, NULL, "transp2_", 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL, NULL); // tr_c2; dims: 32
	conv_02 = nn_append_upsampl_layer(conv_02, 0, NULL, pool_size, true); // 64

	nn_t copy_02 = nn_from_nlop(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(conv_01, 0, NULL)->dims)));

	nn_t combined_02 = nn_combine_FF(conv_02, copy_02); // in: id_c2, cv_c2, C3, tr_c2, cy_02; out: out_c2, out_cy2
	combined_02 = nn_dup_F(combined_02, 0, NULL, 1, NULL); // in: id_c2, cv_c2, C3, tr_c2; out: out_c2, out_cy2
	combined_02 = nn_combine_FF(nn_from_nlop(nlop_zaxpbz_create(5, nn_generic_codomain(conv_01, 0, NULL)->dims, 1.0, 1.0)), combined_02);
	combined_02 = nn_link_F(combined_02, 1, NULL, 0, NULL); // in: out_c2, id_c2, cv_c2, C3, tr_c2; out: out_c2+out_cr2, out_cy2
	combined_02 = nn_link_F(combined_02, 1, NULL, 0, NULL); // in: id_c2, C2; out: out_c2+out_cy2
	conv_01 = nn_chain2_swap_FF(conv_01, 0, NULL, combined_02, 0, NULL); // in: id_c1, cv_c1, C2; out: out_c2+out_cy2

	conv_01 = nn_append_transposed_convcorr_layer(conv_01, 0, NULL, "transp1_", 32, kernel_size, conv, true, PAD_SAME, true, NULL, NULL, NULL); // 64
	conv_01 = nn_append_upsampl_layer(conv_01, 0, NULL, pool_size, true); // 128

	nn_t copy_01 = nn_from_nlop(nlop_from_linop_F(linop_identity_create(N, nn_generic_codomain(network, 0, NULL)->dims))); // create identity network

	nn_t combined_01 = nn_combine_FF(conv_01, copy_01); // in: id_c1, conv_c1, C2, trans_c1, id_cr1; out: out_c1, out_cr1
	combined_01 = nn_dup_F(combined_01, 0, NULL, 1, NULL); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1, out_cr1
	combined_01 = nn_combine_FF(nn_from_nlop(nlop_zaxpbz_create(5, nn_generic_codomain(network, 0, NULL)->dims, 1.0, 1.0)), combined_01); // in: out_c1, out_cr1, ...; out: out_c1+out_cr1, out_c1, out_cr1
	combined_01 = nn_link_F(combined_01, 1, NULL, 0, NULL); // in: out_cr1, id_c1, ...; out: out_c1+out_cr1, out_cr1
	combined_01 = nn_link_F(combined_01, 1, NULL, 0, NULL); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1+out_cr1
	network = nn_chain2_swap_FF(network, 0, NULL, combined_01, 0, NULL); // in: img, conv1, act1, conv2, act2, conv_c1, C2, trans_c1; out: out

	network = nn_append_transposed_convcorr_layer(network, 0, NULL, "transp_", 32, kernel_size, conv, true, PAD_SAME, true, NULL, NULL, NULL); //128
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, MD_BIT(0));
	network = nn_append_transposed_convcorr_layer(network, 0, NULL, "transp_", 32, kernel_size, conv, true, PAD_SAME, true, NULL, NULL, NULL); //128
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, MD_BIT(0));

	network = nn_append_convcorr_layer(network, 0, NULL, "conv_", segm->classes, MAKE_ARRAY(1l, 1l, 1l), conv, PAD_SAME, true, NULL, NULL, NULL);
	// reshape dimensions to {label, x * y * z * batch_size} for softmax activation

	long image_dims[N];
	md_copy_dims(N, image_dims, nn_generic_codomain(network, 0, NULL)->dims);

	long odims[] = {

		nn_generic_codomain(network, 0, NULL)->dims[0],
		md_calc_size(nn_generic_codomain(network, 0, NULL)->N,
		nn_generic_codomain(network, 0, NULL)->dims) / nn_generic_codomain(network, 0, NULL)->dims[0]
	};

	network = nn_reshape_out_F(network, 0, NULL, 2, odims);
	network = nn_append_activation_bias(network, 0, NULL, "dense_bias_", ACT_SOFTMAX, ~MD_BIT(1));

	network = nn_reshape_out_F(network, 0, NULL, N, image_dims);

	return network;
}

/**
 * Create network for semantic segmentation
 *
 * @param N_batch batch size
 */
const struct nlop_s* get_nn_segm(int N_batch, struct segm_s* segm)
{
	unsigned int N = 5;
	long indims[] = { 1, segm->imgx, segm->imgy, 1, N_batch }; // channel, x, y, z, batch
	long expdims[] = { 2, segm->imgx, segm->imgy, 1, N_batch }; // expand dimensions, channel, x, y, z, batch

	const struct nlop_s* network = NULL;
	network = nlop_from_linop_F(linop_expand_create(N, expdims, indims));

	long kernel_size[] = { 3, 3, 1 };
	long pool_size[] = { 2, 2, 1 };
	UNUSED(pool_size);
	bool conv = true;

	// in: [1, x, y, z, b], (channel+channel+1)*conv_kernel*channel

//mw:, 32,32,64,128,256,512,512,transconv, 512,256,128,64,32,32,32
	network = append_convcorr_layer(network, 0, 32, kernel_size, conv, PAD_SAME, true, NULL, NULL); // 128
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_convcorr_layer(network, 0, 32, kernel_size, conv, PAD_SAME, true, NULL, NULL); // 128
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_01 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(network, 0)->dims)); // create avg and upconv network
	conv_01 = append_maxpool_layer(conv_01, 0, pool_size, PAD_VALID, true); // 64
	conv_01 = append_convcorr_layer(conv_01, 0, 64, kernel_size, conv, PAD_SAME, true, NULL, NULL); // cv_c1; 64
	//conv_01 = append_activation_bias(conv_01, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_02 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_01, 0)->dims));
	conv_02 = append_maxpool_layer(conv_02, 0, pool_size, PAD_VALID, true); // dims: 32
	conv_02 = append_convcorr_layer(conv_02, 0, 64, kernel_size, conv, PAD_SAME, true, NULL, NULL); // cv_c2; dims: 32
	//conv_02 = append_activation_bias(conv_02, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_03 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_02, 0)->dims));
	conv_03 = append_maxpool_layer(conv_03, 0, pool_size, PAD_VALID, true); // dims: 16
	conv_03 = append_convcorr_layer(conv_03, 0, 64, kernel_size, conv, PAD_SAME, true, NULL, NULL); // cv_c3; dims: 16
	//conv_03 = append_activation_bias(conv_03, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_04 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_03, 0)->dims));
	conv_04 = append_maxpool_layer(conv_04, 0, pool_size, PAD_VALID, true); // dims: 8
	conv_04 = append_convcorr_layer(conv_04, 0, 64, kernel_size, conv, PAD_SAME, true, NULL, NULL); // cv_c4; dims: 8
	//conv_04 = append_activation_bias(conv_04, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_05 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_04, 0)->dims));
	conv_05 = append_maxpool_layer(conv_05, 0, pool_size, PAD_VALID, true); // dims: 4
	conv_05 = append_convcorr_layer(conv_05, 0, 128, kernel_size, conv, PAD_SAME, true, NULL, NULL); // cv_c5; dims: 4
	//conv_05 = append_activation_bias(conv_05, 0, ACT_RELU, MD_BIT(0));
	conv_05 = append_transposed_convcorr_layer(conv_05, 0, 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL); // tr_c5; dims:4
	//conv_05 = append_activation_bias(conv_05, 0, ACT_RELU, MD_BIT(0));
	conv_05 = append_upsampl_layer(conv_05, 0, pool_size, true); // 8

	const struct nlop_s* copy_05 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_04, 0)->dims));

	const struct nlop_s* combined_05 = nlop_combine_FF(conv_05, copy_05); // in: id_c5, cv_c5, tr_c5, cy_05; out: out_c5, out_cy5
	combined_05 = nlop_dup_F(combined_05, 0,  3); // in: id_c5, cv_c5, tr_c5; out: out_c5, out_cy5
	combined_05 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_04, 0)->dims, 1.0, 1.0), combined_05);
	combined_05 = nlop_link_F(combined_05, 1, 0); // in: out_cy5, id_c5, cv_c5, tr_c5; out: out_c5+out_cy5, out_cy5
	combined_05 = nlop_link_F(combined_05, 1, 0); // in: id_c5, cv_c5, tr_c5; out: out_c5+out_cy5
	conv_04 = nlop_chain2_swap_FF(conv_04, 0, combined_05, 0); // in: id_c4, cv_c4, cv_c5, tr_c5; out: out_c5+out_cy5

	conv_04 = append_transposed_convcorr_layer(conv_04, 0, 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL); // tr_c4; dims:8
	//conv_04 = append_activation_bias(conv_04, 0, ACT_RELU, MD_BIT(0));
	conv_04 = append_upsampl_layer(conv_04, 0, pool_size, true); // 16

	const struct nlop_s* copy_04 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_03, 0)->dims));

	const struct nlop_s* combined_04 = nlop_combine_FF(conv_04, copy_04); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4, cy_4; out: out_c4, out_cy4
	combined_04 = nlop_dup_F(combined_04, 0,  5); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4, out_cy4
	combined_04 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_03, 0)->dims, 1.0, 1.0), combined_04);
	combined_04 = nlop_link_F(combined_04, 1, 0); // in: out_c4, id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4+out_cy4, out_cy4
	combined_04 = nlop_link_F(combined_04, 1, 0); // in: id_c4, C4; out: out_c4+out_cy4
	conv_03 = nlop_chain2_swap_FF(conv_03, 0, combined_04, 0); // in: id_c3, cv_c3, C4; out: out_c4+out_cy4

	conv_03 = append_transposed_convcorr_layer(conv_03, 0, 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL); // tr_c3; dims:8
	//conv_03 = append_activation_bias(conv_03, 0, ACT_RELU, MD_BIT(0));
	conv_03 = append_upsampl_layer(conv_03, 0, pool_size, true); // 32

	const struct nlop_s* copy_03 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_02, 0)->dims));

	const struct nlop_s* combined_03 = nlop_combine_FF(conv_03, copy_03); // in: id_c3, cv_c3, C4, tr_c3, cy_03; out: out_c3, out_cy3
	combined_03 = nlop_dup_F(combined_03, 0,  7); // in: id_c3, cv_c3, C4, tr_c3; out: out1, out2
	combined_03 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_02, 0)->dims, 1.0, 1.0), combined_03);
	combined_03 = nlop_link_F(combined_03, 1, 0); // in: out_c3, id_c3, cv_c3, C4, tr_c3; out: out_c3+out_cy3, out_cy3
	combined_03 = nlop_link_F(combined_03, 1, 0); // in: id_c3, C3; out: out_c3+out_cy3
	conv_02 = nlop_chain2_swap_FF(conv_02, 0, combined_03, 0); // in: id_c2, cv_c2, C3; out: out_c3+out_cy3

	conv_02 = append_transposed_convcorr_layer(conv_02, 0, 64, kernel_size, conv, true, PAD_SAME, true, NULL, NULL); // tr_c2; dims: 32
	//conv_02 = append_activation_bias(conv_02, 0, ACT_RELU, MD_BIT(0));
	conv_02 = append_upsampl_layer(conv_02, 0, pool_size, true); // 64

	const struct nlop_s* copy_02 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_01, 0)->dims));

	const struct nlop_s* combined_02 = nlop_combine_FF(conv_02, copy_02); // in: id_c2, cv_c2, C3, tr_c2, cy_02; out: out_c2, out_cy2
	combined_02 = nlop_dup_F(combined_02, 0,  9); // in: id_c2, cv_c2, C3, tr_c2; out: out_c2, out_cy2
	combined_02 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_01, 0)->dims, 1.0, 1.0), combined_02);
	combined_02 = nlop_link_F(combined_02, 1, 0); // in: out_c2, id_c2, cv_c2, C3, tr_c2; out: out_c2+out_cr2, out_cy2
	combined_02 = nlop_link_F(combined_02, 1, 0); // in: id_c2, C2; out: out_c2+out_cy2
	conv_01 = nlop_chain2_swap_FF(conv_01, 0, combined_02, 0); // in: id_c1, cv_c1, C2; out: out_c2+out_cy2

	conv_01 = append_transposed_convcorr_layer(conv_01, 0, 32,  kernel_size, conv, true, PAD_SAME, true, NULL, NULL); // 64
	//conv_01 = append_activation_bias(conv_01, 0, ACT_RELU, MD_BIT(0));
	conv_01 = append_upsampl_layer(conv_01, 0, pool_size, true); // 128

	const struct nlop_s* copy_01 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(network, 0)->dims)); // create identity network

	const struct nlop_s* combined_01 = nlop_combine_FF(conv_01, copy_01); // in: id_c1, conv_c1, C2, trans_c1, id_cr1; out: out_c1, out_cr1
	combined_01 = nlop_dup_F(combined_01, 0,  11); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1, out_cr1
	combined_01 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(network, 0)->dims, 1.0, 1.0), combined_01); // in: out_c1, out_cr1, ...; out: out_c1+out_cr1, out_c1, out_cr1

	combined_01 = nlop_link_F(combined_01, 1, 0); // in: out_cr1, id_c1, ...; out: out_c1+out_cr1, out_cr1
	combined_01 = nlop_link_F(combined_01, 1, 0); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1+out_cr1
	network = nlop_chain2_swap_FF(network, 0, combined_01, 0); // in: img, conv1, act1, conv2, act2, conv_c1, C2, trans_c1; out: out

	network = append_transposed_convcorr_layer(network, 0, 32, kernel_size, conv, true, PAD_SAME, true, NULL, NULL); //128
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_transposed_convcorr_layer(network, 0, 32, kernel_size, conv, true, PAD_SAME, true, NULL, NULL); //128
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));

	network = append_convcorr_layer(network, 0, segm->classes, MAKE_ARRAY(1l, 1l, 1l), conv, PAD_SAME, true, NULL, NULL); // cv_c4; dims: 8
/*
	const long dump_dims[] = {4, segm->imgx, segm->imgy, 1, N_batch};
	const char dump_char[] = {"Dump_conv/dump_conv"};
	const struct nlop_s* dump_conv = nlop_dump_create(N, dump_dims, dump_char, true, false, false);
	network = nlop_chain2_FF(network, 0, dump_conv, 0);
*/
	// reshape dimensions to {label, x * y * z * batch_size} for softmax activation
	long image_dims[N];
	md_copy_dims(N, image_dims, nlop_generic_codomain(network, 0)->dims);
	long odims[] = {nlop_generic_codomain(network, 0)->dims[0],
			md_calc_size(nlop_generic_codomain(network, 0)->N, nlop_generic_codomain(network, 0)->dims)/nlop_generic_codomain(network, 0)->dims[0]};
	network = nlop_reshape_out_F(network, 0, 2, odims);

	network = append_activation_bias(network, 0, ACT_SOFTMAX, ~MD_BIT(1));

	network = nlop_reshape_out_F(network, 0, N, image_dims);
	//network = nlop_reshape_out_F(network, 0, 2, odims);

	return network;
}

/**
 * Initialise numerical weights if not specified.
 */
int nn_segm_get_num_weights(struct segm_s* segm)
{
	const struct nlop_s* network = get_nn_segm(1, segm);
	//const struct nlop_s* network = get_nn_segm_new(1, segm, STAT_TRAIN)->nlop;
	int result = 0;
	//auto network = get_nn_segm_new(1, segm, STAT_TRAIN);

	network = deflatten_weightsF(network, 1);
	result = nlop_generic_domain(network, 1)->dims[0];

	nlop_free(network);

	return result;
}

nn_weights_t init_nn_segm_new(struct segm_s* segm)
{
	auto network = get_nn_segm_new(1, segm, STAT_TRAIN);
	auto result = nn_weights_create_from_nn(network);

	nn_init(network, result);

	nn_free(network);

	return result;
}

/**
 * Initialise neural network
 *
 * @param weights newly initialised weights
 */
void init_nn_segm(complex float* weights, struct segm_s* segm)
{
#if 1
	const struct nlop_s* network = get_nn_segm(1, segm);

	for (int i = 1; i < nlop_get_nr_in_args(network); i++) {

		const struct iovec_s* tmp = nlop_generic_domain(network, i);

		weights = init_auto(tmp->N, tmp->dims, weights, true);
	}

	nlop_free(network);
#else
	auto network = get_nn_segm_new(1, segm, STAT_TRAIN);
	for (int i = 1; i < nlop_get_nr_in_args(network->nlop); i++){

		const struct iovec_s* tmp = nlop_generic_domain(network->nlop, i);
		weights = init_auto(tmp->N, tmp->dims, weights, true);
	}
	nn_free(network);
#endif
}

void train_nn_segm_new(int N_total, int N_batch, nn_weights_t weights,
		const complex float* in, const complex float* out, long epochs, struct segm_s* segm)
{
	nn_t train = nn_loss_cce_append(get_nn_segm_new(N_batch, segm, STAT_TRAIN), 0, NULL, ~MD_BIT(0));
	nn_debug(DP_INFO, train);

#ifdef USE_CUDA
	if (nn_weights_on_gpu(weights)) {

		auto iov = nlop_generic_domain(nn_get_nlop(train), 0);

		long odims[iov->N];
		md_copy_dims(iov->N, odims, iov->dims);

		odims[iov->N - 1] = N_total;
		out = md_gpu_move(iov->N, odims, out, iov->size);

		iov = nlop_generic_domain(nn_get_nlop(train), 1);

		long idims[iov->N];
		md_copy_dims(iov->N, idims, iov->dims);

		idims[iov->N - 1] = N_total;

		in = md_gpu_move(iov->N, idims, in, iov->size);
	}
#endif

	long NI = nn_get_nr_in_args(train);
	long NO = nn_get_nr_out_args(train);

	assert(NI == 2 + weights->N);

	float* src[NI];
	src[0] = (float*)out;
	src[1] = (float*)in;

	for (int i = 0; i < weights->N; i++)
		src[i + 2] = (float*)weights->tensors[i];

	enum IN_TYPE in_type[NI];
	enum OUT_TYPE out_type[NO];

	nn_get_in_types(train, NI, in_type);
	nn_get_out_types(train, NO, out_type);

	in_type[0] = IN_BATCH;
	in_type[1] = IN_BATCH;

	struct iter6_adam_conf _conf = iter6_adam_conf_defaults;
	_conf.INTERFACE.epochs = epochs;

	iter6_adam(CAST_UP(&_conf),
			nn_get_nlop(train),
			NI, in_type, NULL, src,
			NO, out_type,
			N_batch, N_total / N_batch, NULL, NULL);

	nn_free(train);

#ifdef USE_CUDA
	if (nn_weights_on_gpu(weights)){

		md_free(in);
		md_free(out);
	}
#endif
}
/**
 * Train neural network
 *
 * @param N_batch batch size for one training step
 * @param N_total total number of images
 * @param N_total_val total number of validation images
 * @param weights weights of neural network
 * @param in training images
 * @param out segmentation masks of training images
 * @param in_val validation images
 * @param out_val segmentation masks of validation images
 * @param epochs number of epochs for training
 */
void train_nn_segm(int N_total, int N_batch, int N_total_val, complex float* weights,
		const complex float* in, const complex float* out, long epochs, struct segm_s* segm)
{
	UNUSED(N_total_val);

	const struct nlop_s* network = get_nn_segm(N_batch, segm);

	nlop_debug(DP_INFO, network);

	const struct nlop_s* loss = nlop_cce_create(nlop_generic_codomain(network, 0)->N, nlop_generic_codomain(network, 0)->dims, ~MD_BIT(0));


	const struct nlop_s* nlop_train = nlop_chain2(network, 0, loss, 0);

	nlop_free(loss);
	nlop_free(network);

	float* src[nlop_get_nr_in_args(nlop_train)];
	src[0] = (float*)out;
	src[1] = (float*)in;

	for (int i = 2; i < nlop_get_nr_in_args(nlop_train); i++) {

		src[i] = (float*)weights;
		weights += md_calc_size(nlop_generic_domain(nlop_train, i)->N, nlop_generic_domain(nlop_train, i)->dims);
	}

	long NI = nlop_get_nr_in_args(nlop_train);
	long NO = nlop_get_nr_out_args(nlop_train);

	enum IN_TYPE in_type[NI];

	for (int i = 0; i < NI; i++)
		in_type[i] = IN_OPTIMIZE;

	enum OUT_TYPE out_type[NO];

	for (int o = 0; o < NO; o++)
		out_type[o] = OUT_OPTIMIZE;

	in_type[0] = IN_BATCH;
	in_type[1] = IN_BATCH;

	START_TIMER;
	//struct iter6_adadelta_conf _conf = iter6_adadelta_conf_defaults;
	//struct iter6_sgd_conf _conf = iter6_sgd_conf_defaults;
	struct iter6_adam_conf _conf = iter6_adam_conf_defaults;
	_conf.INTERFACE.epochs = epochs;

	if (NULL == segm->in_val) {

		iter6_adam(CAST_UP(&_conf),
				nlop_train,
				NI, in_type, NULL, src,
				NO, out_type,
				N_batch, N_total / N_batch, NULL, NULL);

	} else {

		float* src_val[nlop_get_nr_in_args(nlop_train)];
		src_val[0] = (float*)(segm->out_val);
		src_val[1] = (float*)(segm->in_val);

		auto nlop_validation = get_nn_segm(N_batch, segm);

		nlop_validation = nlop_chain2_FF(nlop_validation, 0, nlop_cce_create(nlop_generic_codomain(nlop_validation, 0)->N, nlop_generic_codomain(nlop_validation, 0)->dims, ~MD_BIT(0)), 0);
		auto iov0 = nlop_generic_domain(nlop_validation, 0);
		auto iov1 = nlop_generic_domain(nlop_validation, 1);
		auto del0 = nlop_del_out_create(iov0->N, iov0->dims);
		auto del1 = nlop_del_out_create(iov1->N, iov1->dims);

		nlop_validation = nlop_set_input_const_F(nlop_validation, 0, iov0->N, iov0->dims, true, (complex float*)src_val[0]);
		nlop_validation = nlop_set_input_const_F(nlop_validation, 0, iov1->N, iov1->dims, true, (complex float*)src_val[1]);
		nlop_validation = nlop_combine_FF(del1, nlop_validation);
		nlop_validation = nlop_combine_FF(del0, nlop_validation);

		const struct monitor_value_s* monitor_validation_loss = monitor_iter6_nlop_create(nlop_validation, false, 1, (const char*[1]){"val loss"});

		nlop_free(nlop_validation);

		auto monitor = monitor_iter6_create(true, true, 1, MAKE_ARRAY(monitor_validation_loss));

		iter6_adam(CAST_UP(&_conf),
				nlop_train,
				NI, in_type, NULL, src,
				NO, out_type,
				N_batch, N_total / N_batch, NULL, monitor);

		monitor_iter6_dump_record(monitor, segm->val_loss);
		monitor_iter6_free(monitor);
	}

	long dl_tmp = debug_level;

	debug_level = MAX(DP_DEBUG1, debug_level);
	PRINT_TIMER("Trainings");
	debug_level = dl_tmp;

	nlop_free(nlop_train);
}

void predict_nn_segm_new(int N_total, int N_batch, long* prediction, nn_weights_t weights, const complex float* in, struct segm_s* segm)
{
	int loops = 0;
	long N = 5;

	long cfl_dims[] = { segm->imgx, segm->imgy, N_total };
	long img_dims[] = { segm->imgx * segm->imgy * N_total };
	complex float* segm_pred = md_alloc(1, img_dims, CFL_SIZE);

	while (N_total > 0) {

		N_batch = MIN(N_total, N_batch);

		nn_t network = get_nn_segm_new(N_batch, segm, STAT_TEST);
		long odims[N];
		md_copy_dims(N, odims, nn_generic_codomain(network, 0, NULL)->dims);

		long softmax_dims[] = {

			nn_generic_codomain(network, 0, NULL)->dims[0],
			md_calc_size(nn_generic_codomain(network, 0, NULL)->N,
			nn_generic_codomain(network, 0, NULL)->dims) / nn_generic_codomain(network, 0, NULL)->dims[0]
		};

		network = nn_reshape_out_F(network, 0, NULL, 2, softmax_dims);

		while(1 < nn_get_nr_out_args(network))
			network = nn_del_out_F(network, 1, NULL);

		auto nlop_predict = nn_get_nlop_wo_weights(network, weights, false);

		long indims[] = { 1, segm->imgx, segm->imgy, 1, N_batch }; // channel, x, y, z, batch size
		long outdims[] = { segm->classes, (segm->imgx * segm->imgy * N_batch) }; // mask_dim, x * y * z * batch size
		const complex float* tmp_in = in;

#ifdef USE_CUDA
		if (nn_weights_on_gpu(weights)) {

			auto iov = nlop_generic_domain(nlop_predict, 0);
			tmp_in = md_gpu_move(iov->N, iov->dims, in, iov->size);
		}
#endif

		complex float* tmp_out = md_alloc_sameplace(2, outdims, CFL_SIZE, tmp_in);

		nlop_debug(DP_INFO, nlop_predict);

		nlop_apply(nlop_predict, 2, outdims, tmp_out, N, indims, tmp_in);

		nlop_free(nlop_predict);
		nn_free(network);

		complex float* tmp_cpu = md_alloc(2, outdims, CFL_SIZE);
		md_copy(2, outdims, tmp_cpu, tmp_out, CFL_SIZE);

		md_free(tmp_out);

		if (nn_weights_on_gpu(weights))
			md_free(tmp_in);

		hotenc_to_index(N_batch, prediction, tmp_cpu, false, segm);

		for (int i_batch = 0; i_batch < N_batch * segm->imgx * segm->imgy; i_batch++)
			segm_pred[i_batch + loops * N_batch * segm->imgx * segm->imgy] = prediction[i_batch];

		md_free(tmp_cpu);

		prediction += (segm->imgx * segm->imgy * N_batch);

		in += md_calc_size(N, indims);

		N_total -= N_batch;
		loops += 1;
	}

	// create cfl file with image dimensions
	dump_cfl("segmentation_prediction", 3, cfl_dims, segm_pred);

	md_free(segm_pred);
}



/**
 * Creates predictions based on largest probability of encoded indices.
 *
 * @param N_total total number of images
 * @param N_batch batch size
 * @param prediction[(IMG_DIM * IMG_DIM * N_batch)] create prediction for each pixel
 * @param weights trained weights
 */
void predict_nn_segm(int N_total, int N_batch, long* prediction, const complex float* weights, const complex float* in, struct segm_s* segm)
{
	int loops = 0;
	long N = 5;

	long cfl_dims[] = { segm->imgx, segm->imgy, N_total };
	long img_dims[] = { segm->imgx * segm->imgy * N_total };
	complex float* segm_pred = md_alloc(1, img_dims, CFL_SIZE);

	// iterate over each batch in N_total
	while (N_total > 0) {

		N_batch = MIN(N_total, N_batch);

		const struct nlop_s* network = get_nn_segm(N_batch, segm);

		// reshape dims from channel, x, y, z, batch to channel, x * y * z * batch_size for hot encoding
		long odims[N];
		md_copy_dims(N, odims, nlop_generic_codomain(network, 0)->dims);

		long softmax_dims[] = {

			nlop_generic_codomain(network, 0)->dims[0],
			md_calc_size(nlop_generic_codomain(network, 0)->N,
			nlop_generic_codomain(network, 0)->dims) / nlop_generic_codomain(network, 0)->dims[0]
		};

		network = nlop_reshape_out_F(network, 0, 2, softmax_dims);

		while(1 < nlop_get_nr_out_args(network))
			network = nlop_del_out_F(network, 1);

		network = deflatten_weightsF(network, 1);
		network = nlop_set_input_const_F(network, 1, 1, nlop_generic_domain(network, 1)->dims, true, weights);


		long indims[] = { 1, segm->imgx, segm->imgy, 1, N_batch }; // channel, x, y, z, batch size
		long outdims[] = { segm->classes, (segm->imgx * segm->imgy * N_batch) }; // mask_dim, x * y * z * batch size

		complex float* tmp = md_alloc_sameplace(2, outdims, CFL_SIZE, weights);

		nlop_apply(network, 2, outdims, tmp, N, indims, in);

		//nlop_apply(network, 2, outdims, tmp, N, indims, in);
		nlop_free(network);

		complex float* tmp_cpu = md_alloc(2, outdims, CFL_SIZE);
		md_copy(2, outdims, tmp_cpu, tmp, CFL_SIZE);

		hotenc_to_index(N_batch, prediction, tmp_cpu, false, segm);

		for (int i_batch = 0; i_batch < N_batch * segm->imgx * segm->imgy; i_batch++)
			segm_pred[i_batch + loops * N_batch * segm->imgx * segm->imgy] = prediction[i_batch];

		md_free(tmp);
		md_free(tmp_cpu);

		prediction += (segm->imgx * segm->imgy * N_batch);

		in += md_calc_size(N, indims);

		N_total -= N_batch;
		loops += 1;
	}

	// create cfl file with image dimensions
	dump_cfl("segmentation_prediction", 3, cfl_dims, segm_pred);

	md_free(segm_pred);
}



/**
 * Calculate accuracy of trained network
 *
 * @param N_batch batch size
 * @param weights trained weights
 *
 * @return Intersection-over-Union, ratio of Area of Overlap to Area of Union, averaged over all labels
 */
float accuracy_nn_segm_new(int N_total, int N_batch, nn_weights_t weights, const complex float* in, const complex float* out, struct segm_s* segm)
{
	long* prediction = xmalloc(sizeof(long) * segm->imgx * segm->imgy * N_total);// account for stacksize

	predict_nn_segm_new(N_total, N_batch, prediction, weights, in, segm);

	long outdims[] = { segm->classes, segm->imgx, segm->imgy, N_total };

	complex float* tmp_cpu = md_alloc(4, outdims, CFL_SIZE);

	md_copy(4, outdims, tmp_cpu, out, CFL_SIZE);

	long* label = xmalloc(sizeof(long) * segm->imgx * segm->imgy * N_total);

	hotenc_to_index(N_total, label, tmp_cpu, true, segm);

	long* AoO = xmalloc(sizeof(long) * segm->classes);
	long* AoU = xmalloc(sizeof(long) * segm->classes);

	md_clear(1, MD_DIMS(segm->classes), AoO, sizeof(long));
	md_clear(1, MD_DIMS(segm->classes), AoU, sizeof(long));

	// calculate dice coefficient

	for (int i = 0; i < (segm->imgx * segm->imgy * N_total); i++) {

		AoU[(int)label[i]] += 1;
		AoU[(int)prediction[i]] += 1;

		if ((prediction)[i] == (label)[i]) {

			AoO[(int)label[i]] += 1;
			AoU[(int)label[i]] -= 1;
		}
	}

	long* totalNumber = xmalloc(sizeof(long) * segm->classes);

	md_clear(1, MD_DIMS(segm->classes), totalNumber, sizeof(long));

	float dice_coefficient = 0.;

	for (int i = 0; i < (segm->imgx * segm->imgy * N_total); i++) {

		totalNumber[(int)prediction[i]] += 1;
		totalNumber[(int)label[i]] += 1;
	}

	for (int i = 0; i < segm->classes; i++) {

		dice_coefficient += (float)AoO[i] * 2 / ((float) totalNumber[i] * segm->classes);

		printf("Dice coefficient of mask %d = %.6f\n", i,((float)AoO[i] * 2 / (float) totalNumber[i]));
	}

	xfree(label);
	xfree(prediction);

	return dice_coefficient;
/*
	// pixel accuracy
	long num_correct = 0;
	for (int i = 0; i < (IMG_DIM * IMG_DIM * N_total); i++)
		num_correct += (long)(prediction[i] == label[i]);

	return (float)num_correct / (float)(IMG_DIM * IMG_DIM * N_total);

	// Jaccard Index
	long AoO[MASK_DIM] = {};
	long AoU[MASK_DIM] = {};

	for (int i = 0; i < (IMG_DIM * IMG_DIM * N_total); i++){
		AoU[(int)label[i]] += 1;
		AoU[(int)prediction[i]] += 1;
		if ((prediction)[i] == (label)[i]){
			AoO[(int)label[i]] += 1;
			AoU[(int)label[i]] -= 1;
		}
	}
	float IoU = 0.;
	for (int i = 0; i < MASK_DIM; i++){
		IoU += (float)AoO[i] / ((float) AoU[i] * MASK_DIM);
		printf("IoU of mask %d = %.6f\n", i,((float)AoO[i] / ((float) AoU[i])));
	}

	xfree(label);
	xfree(prediction);

	return IoU;
*/
}



/**
 * Calculate accuracy of trained network
 *
 * @param N_batch batch size
 * @param weights trained weights
 *
 * @return Intersection-over-Union, ratio of Area of Overlap to Area of Union, averaged over all labels
 */
float accuracy_nn_segm(int N_total, int N_batch, const complex float* weights, const complex float* in, const complex float* out, struct segm_s* segm)
{
	long* prediction = xmalloc(sizeof(long) * segm->imgx * segm->imgy * N_total);// account for stacksize

	predict_nn_segm(N_total, N_batch, prediction, weights, in, segm);

	long outdims[] = { segm->classes, segm->imgx, segm->imgy, N_total };

	complex float* tmp_cpu = md_alloc(4, outdims, CFL_SIZE);

	md_copy(4, outdims, tmp_cpu, out, CFL_SIZE);

	long* label = xmalloc(sizeof(long) * segm->imgx * segm->imgy * N_total);

	hotenc_to_index(N_total, label, tmp_cpu, true, segm);

	long* AoO = xmalloc(sizeof(long) * segm->classes);
	long* AoU = xmalloc(sizeof(long) * segm->classes);

	md_clear(1, MD_DIMS(segm->classes), AoO, sizeof(long));
	md_clear(1, MD_DIMS(segm->classes), AoU, sizeof(long));

	// calculate dice coefficient

	for (int i = 0; i < (segm->imgx * segm->imgy * N_total); i++) {

		AoU[(int)label[i]] += 1;
		AoU[(int)prediction[i]] += 1;

		if (prediction[i] == label[i]) {

			AoO[(int)label[i]] += 1;
			AoU[(int)label[i]] -= 1;
		}
	}

	long* totalNumber = xmalloc(sizeof(long) * segm->classes);

	md_clear(1, MD_DIMS(segm->classes), totalNumber, sizeof(long));

	float dice_coefficient = 0.;

	for (int i = 0; i < (segm->imgx * segm->imgy * N_total); i++) {

		totalNumber[(int)prediction[i]] += 1;
		totalNumber[(int)label[i]] += 1;
	}

	for (int i = 0; i < segm->classes; i++) {

		dice_coefficient += (float)AoO[i] * 2 / ((float) totalNumber[i] * segm->classes);

		printf("Dice coefficient of mask %d = %.6f\n", i,((float)AoO[i] * 2 / (float) totalNumber[i]));
	}

	xfree(label);
	xfree(prediction);

	return dice_coefficient;
/*
	// pixel accuracy
	long num_correct = 0;
	for (int i = 0; i < (IMG_DIM * IMG_DIM * N_total); i++)
		num_correct += (long)(prediction[i] == label[i]);

	return (float)num_correct / (float)(IMG_DIM * IMG_DIM * N_total);

	// Jaccard Index
	long AoO[MASK_DIM] = {};
	long AoU[MASK_DIM] = {};

	for (int i = 0; i < (IMG_DIM * IMG_DIM * N_total); i++){
		AoU[(int)label[i]] += 1;
		AoU[(int)prediction[i]] += 1;
		if ((prediction)[i] == (label)[i]){
			AoO[(int)label[i]] += 1;
			AoU[(int)label[i]] -= 1;
		}
	}
	float IoU = 0.;
	for (int i = 0; i < MASK_DIM; i++){
		IoU += (float)AoO[i] / ((float) AoU[i] * MASK_DIM);
		printf("IoU of mask %d = %.6f\n", i,((float)AoO[i] / ((float) AoU[i])));
	}

	xfree(label);
	xfree(prediction);

	return IoU;
*/
}
