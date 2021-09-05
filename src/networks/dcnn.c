/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/conv.h"
#include "nlops/const.h"

#include "nn/activation.h"
#include "nn/layers.h"

#include "dcnn.h"




extern void simple_dcnn(const long dims[6], const long krn_dims[6], const complex float* krn, const long bias_dims[6], const complex float* bias, complex float* out, const complex float* in)
{
	unsigned int N = 6;
	unsigned int layers = krn_dims[5];

	assert(krn_dims[3] == krn_dims[4]);
	assert(layers == bias_dims[5]);
	assert(krn_dims[4] == bias_dims[4]);
	assert((1 == dims[3]) && (1 == dims[4]));

	long filters = krn_dims[4];

	long krn_strs[N];
	md_calc_strides(N, krn_strs, krn_dims, CFL_SIZE);

	long bias_strs[N];
	md_calc_strides(N, bias_strs, bias_dims, CFL_SIZE);


	long dims2a[N];
	md_copy_dims(N, dims2a, dims);
	dims2a[3] = krn_dims[3];

	long dims2b[N];
	md_copy_dims(N, dims2b, dims);
	dims2b[4] = krn_dims[4];

	const struct linop_s* lres1 = linop_expand_create(5, dims2a, dims);
	const struct nlop_s* nl = nlop_from_linop(lres1);
	linop_free(lres1);

	struct linop_s* resh = linop_reshape_create(5, dims2a, 5, dims2b);
	struct nlop_s* nresh = nlop_from_linop(resh);
	linop_free(resh);

	for (unsigned int l = 0; l < layers; l++) {

        	long kernel_size[] = {3, 3, 1};
        	long strides[] = {1, 1, 1};
        	long dilation[] = {1, 1, 1};

        	nl = append_convcorr_layer(nl, 0, filters, kernel_size, true, PAD_SAME, false, strides, dilation);
		nl = nlop_chain2_FF(nlop_from_linop_F(linop_transpose_create(5, 3, 4, krn_dims)), 0, nl, nlop_get_nr_in_args(nl) - 1);
		nl = append_activation_bias(nl, 0, (l < layers - 1) ? ACT_RELU : ACT_LIN, MD_BIT(3));

		long bdims[] = {filters};
        	nl = nlop_set_input_const_F(nl, 1, 5, krn_dims, true, krn + l * md_calc_size(5, krn_dims));
        	nl = nlop_set_input_const_F(nl, 1, 1, bdims, true, bias + l * filters);
	}

	nlop_free(nresh);

	const struct linop_s* lres2 = linop_expand_create(5, dims, dims2a);
	nl = nlop_chain_FF(nl, nlop_from_linop(lres2));
	linop_free(lres2);

	debug_printf(DP_INFO, "Applying network...");
	nlop_apply(nl, 5, dims, out, 5, dims, in);
	debug_printf(DP_INFO, " done.\n");
	nlop_free(nl);
}




struct op_dcnn_s {

	INTERFACE(operator_data_t);

	long dims[6];
	long krn_dims[6];
	long bias_dims[6];
	const complex float* krn;
	const complex float* bias;
	float alpha;
};


DEF_TYPEID(op_dcnn_s);

static void op_dcnn_apply(const operator_data_t* _data, float lambda, complex float* dst, const complex float* src)
{
	const struct op_dcnn_s* data = CAST_DOWN(op_dcnn_s, _data);

	// make a copy because dst maybe == src
	complex float* tmp = md_alloc(6, data->dims, CFL_SIZE);
	md_copy(6, data->dims, tmp, src, CFL_SIZE);

	simple_dcnn(data->dims, data->krn_dims, data->krn, data->bias_dims, data->bias, dst, src);

	md_zsmul(6, data->dims, dst, dst, data->alpha / lambda);
	md_zsub(6, data->dims, dst, tmp, dst);

	md_free(tmp);
}

static void op_dcnn_del(const operator_data_t* _data)
{
	const struct op_dcnn_s* data = CAST_DOWN(op_dcnn_s, _data);
	xfree(data);
}

const struct operator_p_s* prox_simple_dcnn_create(unsigned int N, const long dims[6], const long krn_dims[6], const complex float* krn, const long bias_dims[6], const complex float* bias, float alpha)
{
	PTR_ALLOC(struct op_dcnn_s, data);
	SET_TYPEID(op_dcnn_s, data);

	md_copy_dims(6, data->dims, dims);
	md_copy_dims(6, data->krn_dims, krn_dims);
	md_copy_dims(6, data->bias_dims, bias_dims);

	data->krn = krn;
	data->bias = bias;
	data->alpha = alpha;

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), op_dcnn_apply, op_dcnn_del);
}