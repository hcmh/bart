#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "nn/layers.h"
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
#include "iter/batch_gen.h"
#include "iter/monitor_iter6.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/conv.h"

#include "nn/nn_activation.h"
#include "nn/layers_nn.h"
#include "nn/nn_weights.h"
#include "nn/nn_losses.h"

#include "nn/nn.h"

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

static nn_t get_nn_mnist(int N_batch, enum NETWORK_STATUS status)
{
	unsigned int N = 5;
	long indims[] = {1, 28, 28, 1, N_batch};

	nn_t network = nn_from_nlop_F(nlop_from_linop(linop_identity_create(N, indims)));

	long kernel_size[] = {3, 3, 1};
	long pool_size[] = {2, 2, 1};

	bool conv = false;

	network = nn_append_convcorr_layer(network, 0, NULL, "conv_", 32, kernel_size, conv, PAD_VALID, true, NULL, NULL, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, MD_BIT(0));
	network = nn_append_convcorr_layer(network, 0, NULL, "conv_", 64, kernel_size, conv, PAD_VALID, true, NULL, NULL, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "conv_bias_", ACT_RELU, MD_BIT(0));
	network = nn_append_maxpool_layer(network, 0, NULL, pool_size, PAD_VALID, true);

	network = nn_append_flatten_layer(network, 0, NULL);
	network = nn_append_dropout_layer(network, 0, NULL, 0.25, status);
	network = nn_append_dense_layer(network, 0, NULL, "dense_", 128, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "dense_bias_", ACT_RELU, MD_BIT(0));
	network = nn_append_dropout_layer(network, 0, NULL, 0.5, status);
	network = nn_append_dense_layer(network, 0, NULL, "dense_", 10, NULL);
	network = nn_append_activation_bias(network, 0, NULL, "dense_bias_", ACT_SOFTMAX, MD_BIT(0));

	return network;
}

nn_weights_t init_nn_mnist(void)
{
	auto network = get_nn_mnist(1, STAT_TRAIN);
	auto result = nn_weights_create_from_nn(network);
	nn_init(network, result);
	nn_free(network);
	return result;
}

void train_nn_mnist(int N_batch, int N_total, nn_weights_t weights, const complex float* in, const complex float* out, long epochs)
{
	nn_t train = nn_loss_cce_append(get_nn_mnist(N_batch, STAT_TRAIN), 0, NULL);
	nn_debug(DP_INFO, train);

#ifdef USE_CUDA
	if (nn_weights_on_gpu(weights)){

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

	struct iter6_adadelta_conf _conf = iter6_adadelta_conf_defaults;
	_conf.INTERFACE.epochs = epochs;

	iter6_adadelta(CAST_UP(&_conf),
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

void predict_nn_mnist(int N_total, int N_batch, long prediction[N_total], nn_weights_t weights, const complex float* in)
{
	while (N_total > 0) {

		N_batch = MIN(N_total, N_batch);

		nn_t network = get_nn_mnist(N_batch, STAT_TEST);

		auto nlop_predict = nn_get_nlop_wo_weights(network, weights, false);

		long indims[] = {1, 28, 28, 1, N_batch};
		long outdims[] = {10, N_batch};

		const complex float* tmp_in = in;
#ifdef USE_CUDA
		if (nn_weights_on_gpu(weights)){

			auto iov = nlop_generic_domain(nlop_predict, 0);
			tmp_in = md_gpu_move(iov->N, iov->dims, in, iov->size);
		}
#endif

		complex float* tmp_out = md_alloc_sameplace(2, outdims, CFL_SIZE, tmp_in);

		nlop_apply(nlop_predict, 2, outdims, tmp_out, 5, indims, tmp_in);

		nlop_free(nlop_predict);
		nn_free(network);

		complex float* tmp_cpu = md_alloc(2, outdims, CFL_SIZE);
		md_copy(2, outdims, tmp_cpu, tmp_out, CFL_SIZE);

		md_free(tmp_out);
		if (nn_weights_on_gpu(weights))
			md_free(tmp_in);

		hotenc_to_index(N_batch, prediction, 10, tmp_cpu);
		md_free(tmp_cpu);

		prediction += N_batch;
		in += md_calc_size(5, indims);
		N_total -= N_batch;
	}
}

float accuracy_nn_mnist(int N_total, int N_batch, nn_weights_t weights, const complex float* in, const complex float* out)
{
	long prediction[N_total];
	predict_nn_mnist(N_total, N_batch, prediction, weights, in);

	long label[N_total];
	hotenc_to_index(N_total, label, 10, out);

	long num_correct = 0;
	for (int i = 0; i < N_total; i++)
		num_correct += (long)(prediction[i] == label[i]);

	return (float)num_correct / (float)N_total;
}
