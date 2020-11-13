
#include "iter/italgos.h"

#include "num/iovec.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "nn/losses.h"

#include "nn/nn_chain.h"
#include "nn/nn.h"

#include "losses_nn.h"

nn_t nn_loss_mse_append(nn_t network, int o, const char* oname, unsigned long mean_dims)
{
	int nlop_o = nn_get_out_arg_index(network, o, oname);

	auto nlop = nlop_clone(nn_get_nlop(network));
	auto iov = nlop_generic_codomain(nlop, nlop_o);
	nlop = nlop_chain2_swap_FF(nlop, nlop_o, nlop_mse_create(iov->N, iov->dims, mean_dims), 0);
	nlop = nlop_shift_output_F(nlop, nlop_o, 0);

	auto result = nn_from_nlop_F(nlop);

	nn_clone_args(result, network);
	nn_free(network);

	result = nn_shift_input_index_F(result, 0, nn_get_nr_in_args(result) - 1);

	result = nn_set_out_type_F(result, o, oname, OUT_OPTIMIZE);
	return result;
}

nn_t nn_loss_cce_append(nn_t network, int o, const char* oname)
{
	int nlop_o = nn_get_out_arg_index(network, o, oname);

	auto nlop = nlop_clone(nn_get_nlop(network));
	auto iov = nlop_generic_codomain(nlop, nlop_o);
	nlop = nlop_chain2_swap_FF(nlop, nlop_o, nlop_cce_create(iov->N, iov->dims), 0);
	nlop = nlop_shift_output_F(nlop, nlop_o, 0);

	auto result = nn_from_nlop_F(nlop);

	nn_clone_args(result, network);
	nn_free(network);

	result = nn_shift_input_index_F(result, 0, nn_get_nr_in_args(result) - 1);

	result = nn_set_out_type_F(result, o, oname, OUT_OPTIMIZE);
	return result;
}

