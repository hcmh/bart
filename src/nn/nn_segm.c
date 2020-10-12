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

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/conv.h"
#include "nlops/someops.h"

#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/weights.h"
#include "nn/losses.h"
#include "nn/init.h"

#include "nn_segm.h"

/**
 * Hot encode batch
 * (largest value as index for prediction)
 *
 * @param N_batch batch size
 * @param prediction[(IMG_DIM * IMG_DIM * N_batch)] prediction of single pixels for each image in batch
 * @param N_hotenc number of possible predictions
 * @param label indicator if cfl file should be created for label or prediction
 */
static void hotenc_to_index(int N_batch, long prediction[(IMG_DIM * IMG_DIM * N_batch)], int N_hotenc, const complex float* in, bool cfl)
{

	long dims[] = {N_hotenc, (IMG_DIM * IMG_DIM * N_batch)};
	long strs[2];
	md_calc_strides(2, strs, dims, CFL_SIZE);

	complex float* segm;
	for (long i_batch = 0; i_batch < (IMG_DIM * IMG_DIM * N_batch); i_batch++){

		prediction[i_batch] = 0;
		for (int mask_class = 1; mask_class < N_hotenc; mask_class++){

			long pos[] = {mask_class, i_batch};
			long pos_max[] = {prediction[i_batch], i_batch};

			if ((float)MD_ACCESS(2, strs, pos, in) > (float)MD_ACCESS(2, strs, pos_max, in))
				prediction[i_batch] = mask_class;
		}
	}

	long img_dims[] = {IMG_DIM * IMG_DIM * N_batch};
	if (cfl) {

		segm = create_cfl("segmentation_label", 1, img_dims);
		for (int i_batch = 0; i_batch < (IMG_DIM * IMG_DIM * N_batch); i_batch++)
			segm[i_batch] = prediction[i_batch];
		unmap_cfl(1, img_dims, segm);
	}

}

/**
 * Create network for semantic segmentation
 *
 * @param N_batch batch size
 */
const struct nlop_s* get_nn_segm(int N_batch)
{
	unsigned int N = 5;
	long indims[] = {1, IMG_DIM, IMG_DIM, 1, N_batch}; // channel, x, y, z, batch
	long expdims[] = {2, IMG_DIM, IMG_DIM, 1, N_batch}; // expand dimensions, channel, x, y, z, batch

	const struct linop_s* id = linop_expand_create(N, expdims, indims); // optimization assumes nontrivial channeldim, creates second channel filled with zeros
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	long kernel_size[] = {3, 3, 1};
	long pool_size[] = {2, 2, 1};
	UNUSED(pool_size);

	//debug_print_dims(DP_INFO, 5, nlop_generic_codomain(network, 0)->dims);
	//printf("in_args: %d \n", nlop_get_nr_in_args(network));
	//printf("out_args: %d \n", nlop_get_nr_out_args(combined));
	//nlop_debug(DP_INFO, conv_01);
	bool conv = true;
#if 1
	network = append_convcorr_layer(network, 0, 32, kernel_size, conv, PAD_VALID, true, NULL, NULL); // 254
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_convcorr_layer(network, 0, 64, kernel_size, conv, PAD_VALID, true, NULL, NULL); // 252
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_01 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(network, 0)->dims)); // create avg and upconv network
	conv_01 = append_maxpool_layer(conv_01, 0, pool_size, PAD_VALID, true); // 126
	conv_01 = append_convcorr_layer(conv_01, 0, 128, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c1; 124
	//conv_01 = append_activation_bias(conv_01, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_02 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_01, 0)->dims));
	conv_02 = append_maxpool_layer(conv_02, 0, pool_size, PAD_VALID, true); // dims: 62
	conv_02 = append_convcorr_layer(conv_02, 0, 192, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c2; dims: 60
	//conv_02 = append_activation_bias(conv_02, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_03 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_02, 0)->dims));
	conv_03 = append_maxpool_layer(conv_03, 0, pool_size, PAD_VALID, true); // dims: 30
	conv_03 = append_convcorr_layer(conv_03, 0, 256, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c3; dims: 28
	//conv_03 = append_activation_bias(conv_03, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_04 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_03, 0)->dims));
	conv_04 = append_maxpool_layer(conv_04, 0, pool_size, PAD_VALID, true); // dims: 14
	conv_04 = append_convcorr_layer(conv_04, 0, 320, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c4; dims: 12
	//conv_04 = append_activation_bias(conv_04, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_05 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_04, 0)->dims));
	conv_05 = append_maxpool_layer(conv_05, 0, pool_size, PAD_VALID, true); // dims: 6
	conv_05 = append_convcorr_layer(conv_05, 0, 512, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c5; dims: 4
	//conv_05 = append_activation_bias(conv_05, 0, ACT_RELU, MD_BIT(0));
	conv_05 = append_transposed_convcorr_layer(conv_05, 0, 320, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c5; dims:6
	//conv_05 = append_activation_bias(conv_05, 0, ACT_RELU, MD_BIT(0));
	conv_05 = append_upsampl_layer(conv_05, 0, pool_size, true); // 12

	const struct nlop_s* copy_05 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_04, 0)->dims));

	const struct nlop_s* combined_05 = nlop_combine_FF(conv_05, copy_05); // in: id_c5, cv_c5, tr_c5, cy_05; out: out_c5, out_cy5
	combined_05 = nlop_dup_F(combined_05, 0,  3); // in: id_c5, cv_c5, tr_c5; out: out_c5, out_cy5
	combined_05 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_04, 0)->dims, 1.0, 1.0), combined_05);
	combined_05 = nlop_link_F(combined_05, 1, 0); // in: out_cy5, id_c5, cv_c5, tr_c5; out: out_c5+out_cy5, out_cy5
	combined_05 = nlop_link_F(combined_05, 1, 0); // in: id_c5, cv_c5, tr_c5; out: out_c5+out_cy5
	conv_04 = nlop_chain2_swap_FF(conv_04, 0, combined_05, 0); // in: id_c4, cv_c4, cv_c5, tr_c5; out: out_c5+out_cy5

	conv_04 = append_transposed_convcorr_layer(conv_04, 0, 256, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c4; dims:14
	//conv_04 = append_activation_bias(conv_04, 0, ACT_RELU, MD_BIT(0));
	conv_04 = append_upsampl_layer(conv_04, 0, pool_size, true); // 28

	const struct nlop_s* copy_04 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_03, 0)->dims));

	const struct nlop_s* combined_04 = nlop_combine_FF(conv_04, copy_04); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4, cy_4; out: out_c4, out_cy4
	combined_04 = nlop_dup_F(combined_04, 0,  5); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4, out_cy4
	combined_04 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_03, 0)->dims, 1.0, 1.0), combined_04);
	combined_04 = nlop_link_F(combined_04, 1, 0); // in: out_c4, id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4+out_cy4, out_cy4
	combined_04 = nlop_link_F(combined_04, 1, 0); // in: id_c4, C4; out: out_c4+out_cy4
	conv_03 = nlop_chain2_swap_FF(conv_03, 0, combined_04, 0); // in: id_c3, cv_c3, C4; out: out_c4+out_cy4

	conv_03 = append_transposed_convcorr_layer(conv_03, 0, 192, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c3; dims:8
	//conv_03 = append_activation_bias(conv_03, 0, ACT_RELU, MD_BIT(0));
	conv_03 = append_upsampl_layer(conv_03, 0, pool_size, true); // 24

	const struct nlop_s* copy_03 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_02, 0)->dims));

	const struct nlop_s* combined_03 = nlop_combine_FF(conv_03, copy_03); // in: id_c3, cv_c3, C4, tr_c3, cy_03; out: out_c3, out_cy3
	combined_03 = nlop_dup_F(combined_03, 0,  7); // in: id_c3, cv_c3, C4, tr_c3; out: out1, out2
	combined_03 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_02, 0)->dims, 1.0, 1.0), combined_03);
	combined_03 = nlop_link_F(combined_03, 1, 0); // in: out_c3, id_c3, cv_c3, C4, tr_c3; out: out_c3+out_cy3, out_cy3
	combined_03 = nlop_link_F(combined_03, 1, 0); // in: id_c3, C3; out: out_c3+out_cy3
	conv_02 = nlop_chain2_swap_FF(conv_02, 0, combined_03, 0); // in: id_c2, cv_c2, C3; out: out_c3+out_cy3

	conv_02 = append_transposed_convcorr_layer(conv_02, 0, 128, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c2; dims:27
	//conv_02 = append_activation_bias(conv_02, 0, ACT_RELU, MD_BIT(0));
	conv_02 = append_upsampl_layer(conv_02, 0, pool_size, true); // 81

	const struct nlop_s* copy_02 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_01, 0)->dims));

	const struct nlop_s* combined_02 = nlop_combine_FF(conv_02, copy_02); // in: id_c2, cv_c2, C3, tr_c2, cy_02; out: out_c2, out_cy2
	combined_02 = nlop_dup_F(combined_02, 0,  9); // in: id_c2, cv_c2, C3, tr_c2; out: out_c2, out_cy2
	combined_02 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_01, 0)->dims, 1.0, 1.0), combined_02);
	combined_02 = nlop_link_F(combined_02, 1, 0); // in: out_c2, id_c2, cv_c2, C3, tr_c2; out: out_c2+out_cr2, out_cy2
	combined_02 = nlop_link_F(combined_02, 1, 0); // in: id_c2, C2; out: out_c2+out_cy2
	conv_01 = nlop_chain2_swap_FF(conv_01, 0, combined_02, 0); // in: id_c1, cv_c1, C2; out: out_c2+out_cy2

	conv_01 = append_transposed_convcorr_layer(conv_01, 0, 64,  kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // 84
	//conv_01 = append_activation_bias(conv_01, 0, ACT_RELU, MD_BIT(0));
	conv_01 = append_upsampl_layer(conv_01, 0, pool_size, true); //252

	const struct nlop_s* copy_01 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(network, 0)->dims)); // create identity network

	const struct nlop_s* combined_01 = nlop_combine_FF(conv_01, copy_01); // in: id_c1, conv_c1, C2, trans_c1, id_cr1; out: out_c1, out_cr1
	combined_01 = nlop_dup_F(combined_01, 0,  11); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1, out_cr1
	combined_01 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(network, 0)->dims, 1.0, 1.0), combined_01); // in: out_c1, out_cr1, ...; out: out_c1+out_cr1, out_c1, out_cr1

	combined_01 = nlop_link_F(combined_01, 1, 0); // in: out_cr1, id_c1, ...; out: out_c1+out_cr1, out_cr1
	combined_01 = nlop_link_F(combined_01, 1, 0); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1+out_cr1
	network = nlop_chain2_swap_FF(network, 0, combined_01, 0); // in: img, conv1, act1, conv2, act2, conv_c1, C2, trans_c1; out: out
#else
	network = append_convcorr_layer(network, 0, 32, kernel_size, conv, PAD_VALID, true, NULL, NULL); // 254
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_convcorr_layer(network, 0, 32, kernel_size, conv, PAD_VALID, true, NULL, NULL); // 252
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_01 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(network, 0)->dims)); // create avg and upconv network
	conv_01 = append_maxpool_layer(conv_01, 0, pool_size, PAD_VALID, true); // 126
	conv_01 = append_convcorr_layer(conv_01, 0, 64, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c1; 124
	//conv_01 = append_activation_bias(conv_01, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_02 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_01, 0)->dims));
	conv_02 = append_maxpool_layer(conv_02, 0, pool_size, PAD_VALID, true); // dims: 62
	conv_02 = append_convcorr_layer(conv_02, 0, 64, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c2; dims: 60
	//conv_02 = append_activation_bias(conv_02, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_03 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_02, 0)->dims));
	conv_03 = append_maxpool_layer(conv_03, 0, pool_size, PAD_VALID, true); // dims: 30
	conv_03 = append_convcorr_layer(conv_03, 0, 64, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c3; dims: 28
	//conv_03 = append_activation_bias(conv_03, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_04 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_03, 0)->dims));
	conv_04 = append_maxpool_layer(conv_04, 0, pool_size, PAD_VALID, true); // dims: 14
	conv_04 = append_convcorr_layer(conv_04, 0, 64, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c4; dims: 12
	//conv_04 = append_activation_bias(conv_04, 0, ACT_RELU, MD_BIT(0));

	const struct nlop_s* conv_05 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_04, 0)->dims));
	conv_05 = append_maxpool_layer(conv_05, 0, pool_size, PAD_VALID, true); // dims: 6
	conv_05 = append_convcorr_layer(conv_05, 0, 128, kernel_size, conv, PAD_VALID, true, NULL, NULL); // cv_c5; dims: 4
	//conv_05 = append_activation_bias(conv_05, 0, ACT_RELU, MD_BIT(0));
	conv_05 = append_transposed_convcorr_layer(conv_05, 0, 64, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c5; dims:6
	//conv_05 = append_activation_bias(conv_05, 0, ACT_RELU, MD_BIT(0));
	conv_05 = append_upsampl_layer(conv_05, 0, pool_size, true); // 12

	const struct nlop_s* copy_05 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_04, 0)->dims));

	const struct nlop_s* combined_05 = nlop_combine_FF(conv_05, copy_05); // in: id_c5, cv_c5, tr_c5, cy_05; out: out_c5, out_cy5
	combined_05 = nlop_dup_F(combined_05, 0,  3); // in: id_c5, cv_c5, tr_c5; out: out_c5, out_cy5
	combined_05 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_04, 0)->dims, 1.0, 1.0), combined_05);
	combined_05 = nlop_link_F(combined_05, 1, 0); // in: out_cy5, id_c5, cv_c5, tr_c5; out: out_c5+out_cy5, out_cy5
	combined_05 = nlop_link_F(combined_05, 1, 0); // in: id_c5, cv_c5, tr_c5; out: out_c5+out_cy5
	conv_04 = nlop_chain2_swap_FF(conv_04, 0, combined_05, 0); // in: id_c4, cv_c4, cv_c5, tr_c5; out: out_c5+out_cy5

	conv_04 = append_transposed_convcorr_layer(conv_04, 0, 64, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c4; dims:14
	//conv_04 = append_activation_bias(conv_04, 0, ACT_RELU, MD_BIT(0));
	conv_04 = append_upsampl_layer(conv_04, 0, pool_size, true); // 28

	const struct nlop_s* copy_04 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_03, 0)->dims));

	const struct nlop_s* combined_04 = nlop_combine_FF(conv_04, copy_04); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4, cy_4; out: out_c4, out_cy4
	combined_04 = nlop_dup_F(combined_04, 0,  5); // in: id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4, out_cy4
	combined_04 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_03, 0)->dims, 1.0, 1.0), combined_04);
	combined_04 = nlop_link_F(combined_04, 1, 0); // in: out_c4, id_c4, cv_c4, cv_c5, tr_c5, tr_c4; out: out_c4+out_cy4, out_cy4
	combined_04 = nlop_link_F(combined_04, 1, 0); // in: id_c4, C4; out: out_c4+out_cy4
	conv_03 = nlop_chain2_swap_FF(conv_03, 0, combined_04, 0); // in: id_c3, cv_c3, C4; out: out_c4+out_cy4

	conv_03 = append_transposed_convcorr_layer(conv_03, 0, 64, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c3; dims:8
	//conv_03 = append_activation_bias(conv_03, 0, ACT_RELU, MD_BIT(0));
	conv_03 = append_upsampl_layer(conv_03, 0, pool_size, true); // 24

	const struct nlop_s* copy_03 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_02, 0)->dims));

	const struct nlop_s* combined_03 = nlop_combine_FF(conv_03, copy_03); // in: id_c3, cv_c3, C4, tr_c3, cy_03; out: out_c3, out_cy3
	combined_03 = nlop_dup_F(combined_03, 0,  7); // in: id_c3, cv_c3, C4, tr_c3; out: out1, out2
	combined_03 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_02, 0)->dims, 1.0, 1.0), combined_03);
	combined_03 = nlop_link_F(combined_03, 1, 0); // in: out_c3, id_c3, cv_c3, C4, tr_c3; out: out_c3+out_cy3, out_cy3
	combined_03 = nlop_link_F(combined_03, 1, 0); // in: id_c3, C3; out: out_c3+out_cy3
	conv_02 = nlop_chain2_swap_FF(conv_02, 0, combined_03, 0); // in: id_c2, cv_c2, C3; out: out_c3+out_cy3

	conv_02 = append_transposed_convcorr_layer(conv_02, 0, 64, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // tr_c2; dims:27
	//conv_02 = append_activation_bias(conv_02, 0, ACT_RELU, MD_BIT(0));
	conv_02 = append_upsampl_layer(conv_02, 0, pool_size, true); // 81

	const struct nlop_s* copy_02 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(conv_01, 0)->dims));

	const struct nlop_s* combined_02 = nlop_combine_FF(conv_02, copy_02); // in: id_c2, cv_c2, C3, tr_c2, cy_02; out: out_c2, out_cy2
	combined_02 = nlop_dup_F(combined_02, 0,  9); // in: id_c2, cv_c2, C3, tr_c2; out: out_c2, out_cy2
	combined_02 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(conv_01, 0)->dims, 1.0, 1.0), combined_02);
	combined_02 = nlop_link_F(combined_02, 1, 0); // in: out_c2, id_c2, cv_c2, C3, tr_c2; out: out_c2+out_cr2, out_cy2
	combined_02 = nlop_link_F(combined_02, 1, 0); // in: id_c2, C2; out: out_c2+out_cy2
	conv_01 = nlop_chain2_swap_FF(conv_01, 0, combined_02, 0); // in: id_c1, cv_c1, C2; out: out_c2+out_cy2

	conv_01 = append_transposed_convcorr_layer(conv_01, 0, 32,  kernel_size, conv, true, PAD_VALID, true, NULL, NULL); // 84
	//conv_01 = append_activation_bias(conv_01, 0, ACT_RELU, MD_BIT(0));
	conv_01 = append_upsampl_layer(conv_01, 0, pool_size, true); //252

	const struct nlop_s* copy_01 = nlop_from_linop_F(linop_identity_create(N, nlop_generic_codomain(network, 0)->dims)); // create identity network

	const struct nlop_s* combined_01 = nlop_combine_FF(conv_01, copy_01); // in: id_c1, conv_c1, C2, trans_c1, id_cr1; out: out_c1, out_cr1
	combined_01 = nlop_dup_F(combined_01, 0,  11); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1, out_cr1
	combined_01 = nlop_combine_FF(nlop_zaxpbz_create(5, nlop_generic_codomain(network, 0)->dims, 1.0, 1.0), combined_01); // in: out_c1, out_cr1, ...; out: out_c1+out_cr1, out_c1, out_cr1

	combined_01 = nlop_link_F(combined_01, 1, 0); // in: out_cr1, id_c1, ...; out: out_c1+out_cr1, out_cr1
	combined_01 = nlop_link_F(combined_01, 1, 0); // in: id_c1, conv_c1, C2, trans_c1; out: out_c1+out_cr1
	network = nlop_chain2_swap_FF(network, 0, combined_01, 0); // in: img, conv1, act1, conv2, act2, conv_c1, C2, trans_c1; out: out
#endif
	network = append_transposed_convcorr_layer(network, 0, 32, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); //254
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_transposed_convcorr_layer(network, 0, 32, kernel_size, conv, true, PAD_VALID, true, NULL, NULL); //256
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));

	// reshape dimensions to {label, x * y * z * batch_size}
	long odims[] = {nlop_generic_codomain(network, 0)->dims[0],
			md_calc_size(nlop_generic_codomain(network, 0)->N, nlop_generic_codomain(network, 0)->dims)/nlop_generic_codomain(network, 0)->dims[0]};
	network = nlop_reshape_out_F(network, 0, 2, odims);

	//network = append_dropout_layer(network, 0, 0.25);
	network = append_dense_layer(network, 0, MASK_DIM);

	network = append_activation_bias(network, 0, ACT_SOFTMAX, ~MD_BIT(1));

	return network;
}

/**
 * Initialise numerical weights if not specified.
 */
int nn_segm_get_num_weights(void)
{
	const struct nlop_s* network = get_nn_segm(1);
	network = deflatten_weightsF(network, 1);

	int result = (nlop_generic_domain(network, 1)->dims)[0];
	nlop_free(network);
	return result;
}

/**
 * Initialise neural network
 *
 * @param weights newly initialised weights
 */
void init_nn_segm(complex float* weights)
{
	const struct nlop_s* network = get_nn_segm(1);

	for (int i = 0; i < nlop_get_nr_in_args(network); i++){

		const struct iovec_s* tmp = nlop_generic_domain(network, i);
		if (i != 0)
			weights = init_auto(tmp->N, tmp->dims, weights, true);
	}

	nlop_free(network);
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
void train_nn_segm(int N_batch, int N_total, int N_total_val, complex float* weights, const complex float* in, const complex float* out, const complex float* in_val, const complex float* out_val, long epochs)
{
	UNUSED(N_total_val);
	UNUSED(in_val);
	UNUSED(out_val);

	const struct nlop_s* network = get_nn_segm(N_batch);

	const struct nlop_s* loss = nlop_cce_create(nlop_generic_codomain(network, 0)->N, nlop_generic_codomain(network, 0)->dims);
	const struct nlop_s* weighted_loss = nlop_weighted_cce_create(nlop_generic_codomain(network, 0)->N, nlop_generic_codomain(network, 0)->dims, 2l);

	const struct nlop_s* nlop_train = nlop_chain2(network, 0, loss, 0);
	nlop_free(loss);
	nlop_free(weighted_loss);
	nlop_free(network);

	float* src[nlop_get_nr_in_args(nlop_train)];
	src[0] = (float*)out;
	src[1] = (float*)in;
	for (int i = 2; i < nlop_get_nr_in_args(nlop_train); i++){

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

	//if (NULL == in_val)
	//	printf("Validation images will not be compared\n");

	START_TIMER;

	//struct iter6_adadelta_conf _conf = iter6_adadelta_conf_defaults;
	//struct iter6_sgd_conf _conf = iter6_sgd_conf_defaults;
	struct iter6_adam_conf _conf = iter6_adam_conf_defaults;
	_conf.INTERFACE.epochs = epochs;
	iter6_adam(CAST_UP(&_conf),
			nlop_train,
			NI, in_type, NULL, src,
			NO, out_type,
			N_batch, N_total / N_batch, NULL, NULL);
	long dl_tmp = debug_level;
	debug_level = MAX(DP_DEBUG1, debug_level);
	PRINT_TIMER("Trainings");
	debug_level = dl_tmp;
}

/**
 * Creates predictions based on largest probability of encoded indices.
 *
 * @param N_batch batch size
 * @param prediction[(IMG_DIM * IMG_DIM * N_batch)] create prediction for each pixel
 * @param weights trained weights
 */
void predict_nn_segm(int N_total, int N_batch, long prediction[IMG_DIM * IMG_DIM * N_total], const complex float* weights, const complex float* in)
{
	int N_batch_c = N_batch;
	int loops = 0;
	complex float* segm;
	long img_dims[] = {IMG_DIM * IMG_DIM * N_total};
	segm = create_cfl("segmentation_prediction", 1, img_dims);
	while (N_total > 0) {

		N_batch = MIN(N_total, N_batch);

		const struct nlop_s* network = get_nn_segm(N_batch);
		while(1 < nlop_get_nr_out_args(network))
			network = nlop_del_out_F(network, 1);
		network = deflatten_weightsF(network, 1);
		network = nlop_set_input_const_F(network, 1, 1, nlop_generic_domain(network, 1)->dims, true, weights);

		long indims[] = {1, IMG_DIM, IMG_DIM, 1, N_batch}; // channel, x, y, z, batch size
		long outdims[] = {MASK_DIM, (IMG_DIM * IMG_DIM * N_batch)}; // mask_dim, x * y * z * batch size

		complex float* tmp = md_alloc_sameplace(2, outdims, CFL_SIZE, weights);

		nlop_apply(network, 2, outdims, tmp, 5, indims, in);
		nlop_free(network);

		complex float* tmp_cpu = md_alloc(2, outdims, CFL_SIZE);
		md_copy(2, outdims, tmp_cpu, tmp, CFL_SIZE);

		hotenc_to_index(N_batch, prediction, MASK_DIM, tmp_cpu, false);
		for (int i_batch = 0; i_batch < N_batch * IMG_DIM * IMG_DIM; i_batch++)
			segm[i_batch + loops * N_batch_c * IMG_DIM * IMG_DIM] = prediction[i_batch];

		md_free(tmp);
		md_free(tmp_cpu);

		prediction += (IMG_DIM * IMG_DIM * N_batch);
		in += md_calc_size(5, indims);
		N_total -= N_batch;
		loops += 1;
	}
	unmap_cfl(1, img_dims, segm);
}

/**
 * Calculate accuracy of trained network
 *
 * @param N_batch batch size
 * @param weights trained weights
 *
 * @return Intersection-over-Union, ratio of Area of Overlap to Area of Union, averaged over all labels
 */
float accuracy_nn_segm(int N_total, int N_batch, const complex float* weights, const complex float* in, const complex float* out)
{
	long* prediction = xmalloc(sizeof(long) * IMG_DIM * IMG_DIM * N_total);// account for stacksize
	predict_nn_segm(N_total, N_batch, prediction, weights, in);

	long* label = xmalloc(sizeof(long) * IMG_DIM * IMG_DIM * N_total);
	hotenc_to_index(N_total, label, MASK_DIM, out, true);

	long AoO[MASK_DIM] = {};
	long AoU[MASK_DIM] = {};

	// calculate dice coefficient

	for (int i = 0; i < (IMG_DIM * IMG_DIM * N_total); i++){
		AoU[(int)label[i]] += 1;
		AoU[(int)prediction[i]] += 1;
		if ((prediction)[i] == (label)[i]){
			AoO[(int)label[i]] += 1;
			AoU[(int)label[i]] -= 1;
		}
	}
	long totalNumber[MASK_DIM] = {};
	float dice_coefficient = 0.;

	for (int i = 0; i < (IMG_DIM * IMG_DIM * N_total); i++){
		totalNumber[(int)prediction[i]] +=1;
		totalNumber[(int)label[i]] +=1;
	}

	for (int i = 0; i < MASK_DIM; i++){
		dice_coefficient += (float)AoO[i] * 2 / ((float) totalNumber[i] * MASK_DIM);
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
