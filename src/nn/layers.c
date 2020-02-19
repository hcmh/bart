#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/conv.h"

#include "nn_ops.h"
#include "layers.h"

enum NETWORK_STATUS network_status = STAT_TRAIN;

static void perm_shift(int N, int from, int to, int perm[N])
{
	for (int j = 0; j < N; j ++){

		if (j == to){

			perm[j] = from;
			continue;
		}
		int i = j;
		if (j >= from)
			i += 1;
		if (j > to)
			i -= 1;

		perm[j] = i;
	}
}

const struct nlop_s* append_conv_layer(const struct nlop_s* network, int o, int filters, long kernel_size[3], enum CONV_PAD conv_pad, bool channel_first, long strides[3], long dilations[3])
{

	int NO = nlop_get_nr_out_args(network);
	int NI = nlop_get_nr_in_args(network);

	assert(o < NO);
	assert((nlop_generic_codomain(network, o))->N == 5);

	long idims_layer[5]; // channel, x, y, z, batch         or x, y, z, channel, batch
	long odims_layer[5]; // filters, ox, oy, oz, batch         or ox, oy, oz, filters, batch
	long kdims_layer[5]; // filters, channel, kx, ky, kz    or x, y, z channel, filters

	long istrs_layer[5];

	md_copy_dims(5, idims_layer, nlop_generic_codomain(network, o)->dims);
	md_copy_strides(5, istrs_layer, nlop_generic_codomain(network, o)->strs);

	long idims_xyz[3];
	long odims_xyz[3];
	long channels = 0;
	long batch = idims_layer[4];

	if (channel_first){

		channels = idims_layer[0];
		md_copy_dims(3, idims_xyz, idims_layer + 1);
    	} else {

		channels = idims_layer[3];
		md_copy_dims(3, idims_xyz, idims_layer);
    	}

	for (int i = 0; i< 3; i++){

		assert(dilations[i] == 1);//Not supported yet
		assert(strides[i] == 1);
	}

	if (conv_pad == PADDING_VALID){

		for (int i = 0; i < 3; i++)
			odims_xyz[i] = idims_xyz[i] - dilations[i] * (kernel_size[i] - 1);
	} else
		md_copy_dims(3, odims_xyz, idims_xyz);

	long idims_working[6];
	long odims_working[6];
	long kdims_working[6];

	if (channel_first){

		idims_working[0] = 1;
		idims_working[1] = channels;
		md_copy_dims(3, idims_working + 2, idims_xyz);
		idims_working[5] = batch;

		odims_layer[0] = filters;
		md_copy_dims(3, odims_layer + 1, odims_xyz);
		odims_layer[4] = batch;
		odims_working[0] = filters;
		odims_working[1] = 1;
		md_copy_dims(3, odims_working + 2, odims_xyz);
		odims_working[5] = batch;

		kdims_layer[0] = filters;
		kdims_layer[1] = channels;
		md_copy_dims(3, kdims_layer + 2, kernel_size);
		md_copy_dims(5, kdims_working, kdims_layer);
		kdims_working[5] = 1;
	} else {

		md_copy_dims(3, idims_working, idims_xyz);
		idims_working[3] = channels;
		idims_working[4] = 1;
		idims_working[5] = batch;

		md_copy_dims(3, odims_layer, odims_xyz);
		odims_layer[3] = filters;
		odims_layer[4] = batch;

		md_copy_dims(3, odims_working, odims_xyz);
		odims_working[3] = 1;
		odims_working[4] = filters;
		odims_working[5] = batch;

		md_copy_dims(3, kdims_layer, kernel_size);
		kdims_layer[3] = channels;
		kdims_layer[4] = filters;
		md_copy_dims(5, kdims_working, kdims_layer);
		kdims_working[5] = 1;
	}

	const struct nlop_s* conv = nlop_conv_geom_create(6, (channel_first ? 28 : 7), odims_working, idims_working, kdims_working, conv_pad);

	conv = nlop_reshape_out(conv, 0, 5, odims_layer);
	conv = nlop_reshape_in(conv, 0, 5, idims_layer);
	conv = nlop_reshape_in(conv, 1, 5, kdims_layer);
	const struct nlop_s* tmp = nlop_chain2(network, o, conv, 0);
	nlop_free(network);
	nlop_free(conv);

	int perm_in[NI + 1];
	perm_shift(NI + 1, 0, NI, perm_in);
	struct nlop_s* tmp_in = nlop_permute_inputs(tmp, NI + 1, perm_in);
	//nlop_free(tmp);

	int perm_out[NO];
	perm_shift(NO, 0, o, perm_out);
	const struct nlop_s* result = nlop_permute_outputs(tmp_in, NO, perm_out);
	//nlop_free(tmp_in);

	return result;
}

const struct nlop_s* append_maxpool_layer(const struct nlop_s* network, int o, long pool_size[3], enum CONV_PAD conv_pad, bool channel_first)
{
	int NO = nlop_get_nr_out_args(network);
	assert(o < NO);

	assert((nlop_generic_codomain(network, o))->N == 5);

	long idims_layer[5];
	long idims_working[5];
	long pool_size_working[5];

	long idims_xyz[3];

	md_copy_dims(5, idims_layer, nlop_generic_codomain(network, o)->dims);
	md_copy_dims(5, idims_working, nlop_generic_codomain(network, o)->dims);
	md_singleton_dims(5, pool_size_working);
	md_copy_dims(3, pool_size_working + (channel_first ? 1 : 0), pool_size);

	md_copy_dims(3, idims_xyz, idims_layer + (channel_first ? 1 : 0));

	bool resize_needed = false;

	for (int i = (channel_first ? 1 : 0); i < 3 + (channel_first ? 1 : 0); i++){

		if (idims_working[i] % pool_size_working[i]== 0)
			continue;

		resize_needed = true;

		if (conv_pad == PADDING_VALID){

			idims_working[i] -= (idims_working[i] % pool_size_working[i]);
			continue;
		}

		if (conv_pad == PADDING_VALID){

			idims_working[i] += pool_size_working[i] - (idims_working[i] % pool_size_working[i]);
			continue;
		}

		assert(0);
	}

	const struct nlop_s* pool_op = nlop_maxpool_create(5, idims_working, pool_size_working);

	if (resize_needed){

		struct linop_s* lin_res = linop_expand_create(5, idims_layer, idims_working);
		struct nlop_s* nlop_res = nlop_from_linop(lin_res);
		linop_free(lin_res);
		pool_op = nlop_chain_FF(nlop_res, pool_op);
    	}

	struct nlop_s* tmp = nlop_chain2(network, o, pool_op, 0);
	nlop_free(network);
	nlop_free(pool_op);

	int perm_out[NO];
	perm_shift(NO, 0, o, perm_out);
	struct nlop_s* result = nlop_permute_outputs(tmp, NO, perm_out);
	nlop_free(tmp);

	return result;
}


const struct nlop_s* append_dense_layer(const struct nlop_s* network, int o, int out_neurons)
{

	int NO = nlop_get_nr_out_args(network);
	int NI = nlop_get_nr_in_args(network);

	assert(o < NO);
	assert((nlop_generic_codomain(network, o))->N == 2);

	long batch = (nlop_generic_codomain(network, o)->dims)[1];
	long in_neurons = (nlop_generic_codomain(network, o)->dims)[0];

	long idims_layer[] = {in_neurons, batch};       // in neurons, batch
	long odims_layer[] = {out_neurons, batch};      // out neurons, batch
	long wdims_layer[] = {out_neurons, in_neurons}; // out neurons, in neurons

	long istrs_layer[2];
	md_copy_strides(2, istrs_layer, nlop_generic_codomain(network, o)->strs);

	long idims_working[] = {1, in_neurons, batch};       // in neurons, batch
	long odims_working[] = {out_neurons, 1, batch};      // out neurons, batch
	long wdims_working[] = {out_neurons, in_neurons, 1}; // out neurons, in neurons

	const struct nlop_s* matmul = nlop_tenmul_create(3, odims_working, idims_working, wdims_working);
	matmul = nlop_reshape_out(matmul, 0, 2, odims_layer);
	matmul = nlop_reshape_in(matmul, 0, 2, idims_layer);
	matmul = nlop_reshape_in(matmul, 1, 2, wdims_layer);

	const struct nlop_s* tmp = nlop_chain2(network, o, matmul, 0);
	nlop_free(network);
	nlop_free(matmul);

	int perm_in[NI + 1];
	perm_shift(NI + 1, 0, NI, perm_in);
	int perm_out[NO];
	perm_shift(NO, 0, o, perm_out);

	struct nlop_s* tmp_in = nlop_permute_inputs(tmp, NI + 1, perm_in);
	nlop_free(tmp);
	struct nlop_s* result = nlop_permute_outputs(tmp_in, NO, perm_out);
	nlop_free(tmp_in);

	return result;
}

const struct nlop_s* append_dropout_layer(const struct nlop_s* network, int o, float p)
{
	int NO = nlop_get_nr_out_args(network);
	//int NI = nlop_get_nr_in_args(network);

	assert(o < NO);

	unsigned int N = nlop_generic_codomain(network, o)->N;
	long idims[N];
	md_copy_dims(N, idims, nlop_generic_codomain(network, o)->dims);

	const struct nlop_s* dropout_op = nlop_dropout_create(N, idims, p, 0);
	const struct nlop_s* tmp = nlop_chain2(network, o, dropout_op, 0);
	nlop_free(network);
	nlop_free(dropout_op);

	int perm_out[NO];
	perm_shift(NO, 0, o, perm_out);
	struct nlop_s* result = nlop_permute_outputs(tmp, NO, perm_out);
	nlop_free(tmp);

	return result;
}

const struct nlop_s* append_flatten_layer(const struct nlop_s* network, int o)
{
	int NO = nlop_get_nr_out_args(network);
	//int NI = nlop_get_nr_in_args(network);
	assert(o < NO);

	unsigned int N = nlop_generic_codomain(network, o)->N;

	long idims[N];
	md_copy_dims(N, idims, nlop_generic_codomain(network, o)->dims);

	long size = md_calc_size(N - 1, idims);
	long odims[] = {size, idims[N - 1]};

	return nlop_reshape_out(network, o, 2, odims);
}
