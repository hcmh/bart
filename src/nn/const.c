#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "iter/italgos.h"
#include "misc/misc.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"

#include "nn/nn.h"
#include "const.h"


nn_t nn_set_input_const_F(nn_t op, int i, const char* iname, int N, const long dims[N], _Bool copy, const _Complex float* in)
{
	i = nn_get_in_arg_index(op, i, iname);
	auto result = nn_from_nlop_F(nlop_set_input_const(nn_get_nlop(op), i, N, dims, copy, in));

	for (unsigned int j = 0, jp = 0; j < nn_get_nr_in_args(result); j++) {

		if (i == (int)j) jp++;
		nn_clone_arg_i_from_i(result, j, op, jp);
		jp++;
	}

	for (unsigned int j = 0; j < nn_get_nr_out_args(result); j++)
		nn_clone_arg_o_from_o(result, j, op, j);

	nn_free(op);

	return result;
}

nn_t nn_del_out_F(nn_t op, int o, const char* oname)
{
	o = nn_get_out_arg_index(op, o, oname);
	auto result = nn_from_nlop_F(nlop_del_out(nn_get_nlop(op), o));

	for (unsigned int j = 0, jp = 0; j < nn_get_nr_out_args(result); j++) {

		if (o == (int)j) jp++;
		nn_clone_arg_o_from_o(result, j, op, jp);
		jp++;
	}

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);

	nn_free(op);

	return result;
}

nn_t nn_del_out_bn_F(nn_t op)
{
	auto result = op;
	for (int o = nn_get_nr_out_args(op) - 1; o >= 0; o--)
		if (OUT_BATCHNORM == result->out_types[o])
			result = nn_del_out_F(result, nn_get_out_index_from_arg_index(result, o), nn_get_out_name_from_arg_index(result, o));

	return result;
}

nn_t nn_ignore_input_F(nn_t op, int i, const char* iname, int N, const long dims[N], _Bool copy, const _Complex float* in)
{
	i = nn_get_in_arg_index(op, i, iname);
	auto nlop = nlop_set_input_const(nn_get_nlop(op), i, N, dims, copy, in);
	nlop = nlop_combine_FF(nlop_del_out_create(N, dims), nlop);
	nlop = nlop_shift_input_F(nlop, i, 0);
	auto result = nn_from_nlop_F(nlop);

	for (unsigned int j = 0; j < nn_get_nr_in_args(result); j++)
		nn_clone_arg_i_from_i(result, j, op, j);

	for (unsigned int j = 0; j < nn_get_nr_out_args(result); j++)
		nn_clone_arg_o_from_o(result, j, op, j);

	nn_free(op);

	return result;
}

