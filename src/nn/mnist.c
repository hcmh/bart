
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

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

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

#include "nn/activation.h"
#include "nn/layers.h"
#include "nn/weights.h"
#include "nn/nn.h"
#include "nn/losses.h"
#include "nn/init.h"

#include "mnist.h"

static void hotenc_to_index(int N_batch, long prediction[N_batch], int N_hotenc, const complex float* in)
{

	long dims[] = {N_hotenc, N_batch};
    	long strs[2];
    	md_calc_strides(2, strs, dims, CFL_SIZE);

    	for (int i_batch = 0; i_batch < N_batch; i_batch++){

		prediction[i_batch] = 0;
		for (int i = 1; i < N_hotenc; i++){

        		long pos[] = {i, i_batch};
            		long pos_max[] = {prediction[i_batch], i_batch};

        		if ((float)MD_ACCESS(2, strs, pos, in) > (float)MD_ACCESS(2, strs, pos_max, in))
                		prediction[i_batch] = i;
        	}
    	}
}

const struct nlop_s* get_nn_mnist(int N_batch)
{
	unsigned int N = 5;
	long indims[] = {1, 28, 28, 1, N_batch};
	long expdims[] = {2, 28, 28, 1, N_batch};

	const struct linop_s* id = linop_expand_create(N, expdims, indims); // optimization assumes nontrivial channeldim
	const struct nlop_s* network = nlop_from_linop(id);
	linop_free(id);

	long kernel_size[] = {3, 3, 1};
    	long pool_size[] = {2, 2, 1};

	bool conv = false;

	network = append_convcorr_layer(network, 0, 32, kernel_size, conv, PAD_VALID, true, NULL, NULL);
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_convcorr_layer(network, 0, 64, kernel_size, conv, PAD_VALID, true, NULL, NULL);
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_maxpool_layer(network, 0, pool_size, PAD_VALID, true);

	network = append_flatten_layer(network, 0);
	network = append_dropout_layer(network, 0, 0.25);
	network = append_dense_layer(network, 0, 128);
	network = append_activation_bias(network, 0, ACT_RELU, MD_BIT(0));
	network = append_dropout_layer(network, 0, 0.5);
	network = append_dense_layer(network, 0, 10);
	network = append_activation_bias(network, 0, ACT_SOFTMAX, MD_BIT(0));

	return network;
}

int nn_mnist_get_num_weights(void)
{
	const struct nlop_s* network = get_nn_mnist(1);
	network = deflatten_weightsF(network, 1);

	int result = (nlop_generic_domain(network, 1)->dims)[0];
	nlop_free(network);
	return result;
}


void init_nn_mnist(complex float* weights)
{
	const struct nlop_s* network = get_nn_mnist(1);

	for (int i = 0; i < nlop_get_nr_in_args(network); i++){

		const struct iovec_s* tmp = nlop_generic_domain(network, i);
		if (i != 0)
            		weights = init_auto(tmp->N, tmp->dims, weights, true);
	}

	nlop_free(network);
}

void train_nn_mnist(int N_batch, int N_total, complex float* weights, const complex float* in, const complex float* out, long epochs)
{
	network_status = STAT_TRAIN;

	const struct nlop_s* network = get_nn_mnist(N_batch);
	const struct nlop_s* loss = nlop_cce_create(nlop_generic_codomain(network, 0)->N, nlop_generic_codomain(network, 0)->dims);

	const struct nlop_s* nlop_train = nlop_chain2_FF(network, 0, loss, 0);

	float* src[nlop_get_nr_in_args(nlop_train)];
	src[0] = (float*)out;
	src[1] = (float*)in;
	for (int i = 2; i < nlop_get_nr_in_args(nlop_train); i++){

		src[i] =(float*)weights;
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

	struct iter6_adadelta_conf _conf = iter6_adadelta_conf_defaults;
	_conf.epochs = epochs;

	iter6_adadelta(CAST_UP(&_conf),
			nlop_train,
			NI, in_type, src,
			NO, out_type,
			N_batch, N_total);

	nlop_free(nlop_train);
}

void predict_nn_mnist(int N_batch, long prediction[N_batch], const complex float* weights, const complex float* in)
{
	network_status = STAT_TEST;

	const struct nlop_s* network = get_nn_mnist(N_batch);
	network = deflatten_weightsF(network, 1);
	network = nlop_set_input_const_F(network, 1, 1, nlop_generic_domain(network, 1)->dims, weights);

	long indims[] = {1, 28, 28, 1, N_batch};
	long outdims[] = {10, N_batch};

	complex float* tmp = md_alloc_sameplace(2, outdims, CFL_SIZE, weights);

	nlop_apply(network, 2, outdims, tmp, 5, indims, in);
	nlop_free(network);

	hotenc_to_index(N_batch, prediction, 10, tmp);

	md_free(tmp);
}

float accuracy_nn_mnist(int N_batch, const complex float* weights, const complex float* in, const complex float* out)
{
	long prediction[N_batch];
	predict_nn_mnist(N_batch, prediction, weights, in);

	long label[N_batch];
	hotenc_to_index(N_batch, label, 10, out);

	long num_correct = 0;
    	for (int i = 0; i < N_batch; i++)
        	num_correct += (long)(prediction[i] == label[i]);

    	return (float)num_correct / (float)N_batch;
}
