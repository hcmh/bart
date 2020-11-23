#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "misc/misc.h"

#include "nn/init.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "nn/nn.h"
#include "chain.h"

/**
 * Reshape output of nn_t
 *
 * @param op nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param N number of new output dimensions
 * @param odims new output dimensions
 *
 * @returns nn_t with reshaped output
 */
nn_t nn_reshape_out(nn_t op, int o, const char* oname, int N, const long odims[N])
{
	o = nn_get_out_arg_index(op, o, oname);
	auto result = nn_from_nlop_F(nlop_reshape_out(nn_get_nlop(op), o, N, odims));

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (unsigned int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	return result;
}

/**
 * Reshape input of nn_t
 *
 * @param op nn_t struct
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @param N number of new input dimensions
 * @param idims new input dimensions
 *
 * @returns nn_t with reshaped input
 */
nn_t nn_reshape_in(nn_t op, int i, const char* iname, int N, const long idims[N])
{
	i = nn_get_in_arg_index(op, i, iname);
	auto result = nn_from_nlop_F(nlop_reshape_in(nn_get_nlop(op), i, N, idims));

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (unsigned int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	auto iov = nlop_generic_domain(op->nlop, i);
	auto init_tmp = init_reshape_create(iov->N, iov->dims, result->initializers[i]);
	initializer_free(result->initializers[i]);
	result->initializers[i] = init_tmp;

	return result;
}

/**
 * Reshape output of nn_t and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param N number of new output dimensions
 * @param odims new output dimensions
 * @param clear clear acquired k-space
 *
 * @returns nn_t with reshaped output
 */
nn_t nn_reshape_out_F(nn_t op, int o, const char* oname, int NO, const long odims[NO])
{
	auto result = nn_reshape_out(op, o, oname, NO, odims);
	nn_free(op);
	return result;
}

/**
 * Reshape input of nn_t and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 * @param N number of new input dimensions
 * @param idims new input dimensions
 *
 * @returns nn_t with reshaped input
 */
nn_t nn_reshape_in_F(nn_t op, int i, const char* iname, int NI, const long idims[NI])
{
	auto result = nn_reshape_in(op, i, iname, NI, idims);
	nn_free(op);
	return result;
}

/**
 * Reshape input of nn_t to have an additional singleton dimension and free nn_t
 * {dims[0], ..., dims[N-1]} -> {dims[0], ..., dims[N-1], 1}
 *
 * @param op nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns nn_t with reshaped input
 */
nn_t nn_append_singleton_dim_in_F(nn_t op, int i, const char* iname)
{
	auto iov = nn_generic_domain(op, i, iname);
	long dims[iov->N + 1];
	md_copy_dims(iov->N, dims, iov->dims);
	dims[iov->N] = 1;

	return nn_reshape_in_F(op, i, iname, iov->N + 1, dims);
}

/**
 * Reshape output of nn_t to have an additional singleton dimension and free nn_t
 * {dims[0], ..., dims[N-1]} -> {dims[0], ..., dims[N-1], 1}
 *
 * @param op nn_t struct (will be freed)
 * @param o input index (ignored if oname != NULL)
 * @param oname name of output
 *
 * @returns nn_t with reshaped output
 */
nn_t nn_append_singleton_dim_out_F(nn_t op, int o, const char* oname)
{
	auto iov = nn_generic_codomain(op, o, oname);
	long dims[iov->N + 1];
	md_copy_dims(iov->N, dims, iov->dims);
	dims[iov->N] = 1;

	return nn_reshape_out_F(op, o, oname, iov->N + 1, dims);
}

/**
 * Permute inputs of nn_t
 * Input i of result = input perm[i] of op
 *
 * @param op nn_t struct
 * @param I2 no. of inputs of op (either total or unnamed)
 * @param perm permutation of inputs
 *
 * @returns nn_t with permuted inputs
 *
 * @note If I2 equals the number of unnamed inputs, only the unnamed inputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_inputs(nn_t op, unsigned int I2, const int perm[I2])
{
	assert((nn_get_nr_in_args(op) == I2) || (nn_get_nr_unnamed_in_args(op) == I2));

	unsigned int II = nn_get_nr_in_args(op);
	int nperm[II];

	for (unsigned int i = 0; i < II; i++)
		nperm[i] = i;

	if (nn_get_nr_unnamed_in_args(op) == I2) {

		for(unsigned int i = 0; i < nn_get_nr_unnamed_in_args(op); i++)
			nperm[nn_get_in_arg_index(op, i, NULL)] = nn_get_in_arg_index(op, perm[i], NULL);
	} else {

		for(unsigned int i = 0; i < nn_get_nr_in_args(op); i++)
			nperm[i] = perm[i];
	}

	auto result = nn_from_nlop_F(nlop_permute_inputs(nn_get_nlop(op), II, nperm));

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, nperm[i]);
	for (unsigned int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	return result;
}

/**
 * Permute outputs of nn_t
 * Output o of result = output perm[o] of op
 *
 * @param op nn_t struct
 * @param O2 no. of outputs of op (either total or unnamed)
 * @param perm permutation of outputs
 *
 * @returns nn_t with permuted outputs
 *
 * @note If O2 equals the number of unnamed outputs, only the unnamed outputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_outputs(nn_t op, unsigned int O2, const int perm[O2])
{
	assert((nn_get_nr_out_args(op) == O2) || (nn_get_nr_unnamed_out_args(op) == O2));

	unsigned int OO = nn_get_nr_out_args(op);
	int nperm[OO];

	for (unsigned int i = 0; i < OO; i++)
		nperm[i] = i;

	if (nn_get_nr_unnamed_out_args(op) == O2) {

		for(unsigned int i = 0; i < nn_get_nr_unnamed_out_args(op); i++)
			nperm[nn_get_out_arg_index(op, i, NULL)] = nn_get_out_arg_index(op, perm[i], NULL);
	} else {

		for(unsigned int i = 0; i < nn_get_nr_out_args(op); i++)
			nperm[i] = perm[i];
	}

	auto result = nn_from_nlop_F(nlop_permute_outputs(nn_get_nlop(op), OO, nperm));

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (unsigned int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, nperm[i]);

	return result;
}

/**
 * Permute inputs of nn_t and frees nn_t
 * Input i of result = input perm[i] of op
 *
 * @param op nn_t struct (will be freed)
 * @param I2 no of inputs (either total or unnamed)
 * @param perm permutation of inputs
 *
 * @returns nn_t with permuted inputs
 *
 * @note If I2 equals the number of unnamed inputs, only the unnamed inputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_inputs_F(nn_t op, unsigned int I2, const int perm[I2])
{
	auto result = nn_permute_inputs(op, I2, perm);
	nn_free(op);
	return result;
}

/**
 * Permute outputs of nn_t and frees nn_t
 * Output o of result = output perm[o] of op
 *
 * @param op nn_t struct (will be freed)
 * @param O2 no. of outputs of op (either total or unnamed)
 * @param perm permutation of outputs
 *
 * @returns nn_t with permuted outputs
 *
 * @note If O2 equals the number of unnamed outputs, only the unnamed outputs are permuted, i.e. the named ones stay at their positions
 */
nn_t nn_permute_outputs_F(nn_t op, unsigned int O2, const int perm[O2])
{
	auto result = nn_permute_outputs(op, O2, perm);
	nn_free(op);
	return result;
}

/**
 * Combine two nn_t's to one.
 *
 * The resulting nn_t will have a total amount of in- and outputs corresponding to the respective sum of in- and outputs of a and b.
 * The first in-/outputs correspond to a, the latter ones to b.
 * When the resulting nn_t is applied, the apply function of b is called first and the one of a afterwards, thus, if inputs of b depend on outputs of a, the result is undetermined.
 * a and b must not have mutual input names or output names
 *
 * Inputs:	ia1 ia2 ia3 ib1 ib2 ib3
 *		 |   |   |   |   |   |
 * Nlops:	(    a    ) (    b    )
 *		 |   |       |   |   |
 * Outputs:	oa1 oa2     ob1 ob2 ob3
 *
 * @param a nn_t struct
 * @param b nn_t struct
 *
 * @returns combined nn_t (a, b)
 */
nn_t nn_combine(nn_t a, nn_t b)
{
	unsigned int IIa = nn_get_nr_in_args(a);
	unsigned int IIb = nn_get_nr_in_args(b);
	unsigned int OOa = nn_get_nr_out_args(a);
	unsigned int OOb = nn_get_nr_out_args(b);

	for (unsigned int ia = 0; ia < IIa; ia++)
		for (unsigned int ib = 0; ib < IIb; ib++)
			assert(    (NULL == nn_get_in_names(a)[ia])
				|| (NULL == nn_get_in_names(b)[ib])
				|| (0 != strcmp(nn_get_in_names(a)[ia], nn_get_in_names(b)[ib])));

	for (unsigned int ia = 0; ia < OOa; ia++)
		for (unsigned int ib = 0; ib < OOb; ib++)
			assert(    (NULL == nn_get_out_names(a)[ia])
				|| (NULL == nn_get_out_names(b)[ib])
				|| (0 != strcmp(nn_get_out_names(a)[ia], nn_get_out_names(b)[ib])));

	auto result = nn_from_nlop_F(nlop_combine(nn_get_nlop(a), nn_get_nlop(b)));

	for (unsigned int i = 0; i < IIa; i++)
		nn_clone_arg_i_from_i(result, i, a, i);
	for (unsigned int i = 0; i < IIb; i++)
		nn_clone_arg_i_from_i(result, IIa + i, b, i);

	for (unsigned int i = 0; i < OOa; i++)
		nn_clone_arg_o_from_o(result, i, a, i);
	for (unsigned int i = 0; i < OOb; i++)
		nn_clone_arg_o_from_o(result, OOa + i, b, i);

	return result;
}

/**
 * Combine two nn_t's to one and free the former ones
 *
 * The resulting nn_t will have a total amount of in- and outputs corresponding to the respective sum of in- and outputs of a and b.
 * The first in-/outputs correspond to a, the latter ones to b.
 * When the resulting nn_t is applied, the apply function of b is called first and the one of a afterwards, thus, if inputs of b depend on outputs of a, the result is undetermined.
 * a and b must not have mutual input names or output names
 *
 * Inputs:	ia1 ia2 ia3 ib1 ib2 ib3
 *		 |   |   |   |   |   |
 * Nlops:	(    a    ) (    b    )
 *		 |   |       |   |   |
 * Outputs:	oa1 oa2     ob1 ob2 ob3
 *
 * @param a nn_t struct (will be freed)
 * @param b nn_t struct (will be freed)
 *
 * @returns combined nn_t (a, b)
 */
nn_t nn_combine_FF(nn_t a, nn_t b)
{
	auto result = nn_combine(a, b);
	nn_free(a);
	nn_free(b);
	return result;
}

/**
 * Link an output of nn_t op in one of its inputs
 *
 * The output must be computed before the input is accessed (c.f. nn_combine).
 *
 *		  |  |
 * Inputs:	 i1 i2 i3--<-
 * Nlops:	(    op   ) |
 * Outputs:	 o1 o2-->---
 *		  |
 *
 * @param op nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns linked nn_t
 */
nn_t nn_link(nn_t op, int o, const char* oname, int i, const char* iname)
{
	unsigned int OO = nn_get_nr_out_args(op) - 1;
	unsigned int II = nn_get_nr_in_args(op) - 1;

	o = nn_get_out_arg_index(op, o, oname);
	i = nn_get_in_arg_index(op, i, iname);

	auto result = nn_from_nlop_F(nlop_link(nn_get_nlop(op), o, i));

	for (unsigned int ii = 0, ip = 0; ii < II; ii++){

		if ((int)ii == i)
			ip++;
		nn_clone_arg_i_from_i(result, ii, op, ip);
		ip++;
	}

	for (unsigned int ii = 0, ip = 0; ii < OO; ii++){

		if ((int)ii == o)
			ip++;
		nn_clone_arg_o_from_o(result, ii, op, ip);
		op++;
	}

	return result;
}

/**
 * Link an output of nn_t in one of its inputs and frees nn_t
 *
 * The output must be computed before the input is accessed (c.f. nn_combine).
 *
 *		  |  |
 * Inputs:	 i1 i2 i3--<-
 * Nlops:	(    op   ) |
 * Outputs:	 o1 o2-->---
 *		  |
 *
 * @param op nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns linked nn_t
 */
nn_t nn_link_F(nn_t x, int o, const char* oname, int i, const char* iname)
{
	auto result = nn_link(x, o, oname, i, iname);
	nn_free(x);
	return result;
}

/**
 * Chain an output of nn_t a in an input of nn_t b
 *
 * This is a combination of nn_combine and nn_link.
 * The first inputs of the resulting nn_t correspond to the inputs of b (except the input i), the latter ones to a.
 * The first outputs of the resulting nn_t correspond to the outputs of b, the latter ones to a, (except the output o).
 *
 * @param a nn_t struct
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param b nn_t struct
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int OO = nn_get_nr_unnamed_out_args(b);
	return nn_link_F(nn_combine(b, a), (NULL == oname) ? o + OO : 0, oname, i, iname);
}

/**
 * Chain an output of nn_t a in an input of nn_t b and free both nn_ts
 *
 * This is a combination of nn_combine and nn_link.
 * The first inputs of the resulting nn_t correspond to the inputs of b (except the input i), the latter ones to a.
 * The first outputs of the resulting nn_t correspond to the outputs of b, the latter ones to a, (except the output o).
 *
 * @param a nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int OO = nn_get_nr_unnamed_out_args(b);
	return nn_link_F(nn_combine_FF(b, a), (NULL == oname) ? o + OO : 0, oname, i, iname);
}

/**
 * Chain an output of nn_t a in an input of nn_t b, permute the inputs of a to the end and free both nn_ts
 *
 * This is a combination of nn_combine and nn_link and nn_permute_inputs.
 * The first inputs of the resulting nn_t correspond to the inputs of a, the latter ones to b (except the input i).
 * The first outputs of the resulting nn_t correspond to the outputs of b, the latter ones to a, (except the output o).
 *
 * @param a nn_t struct (will be freed)
 * @param o output index (ignored if oname != NULL)
 * @param oname name of output
 * @param nn_t struct (will be freed)
 * @param i input index (ignored if iname != NULL)
 * @param iname name of input
 *
 * @returns chained nn_t
 */
nn_t nn_chain2_swap_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int unIIa = nn_get_nr_in_args(a);

	auto result = nn_chain2_FF(a, o, oname, b, i, iname);

	int unII = nn_get_nr_in_args(result);
	int perm[unII];

	for (int i = 0; i < unII; i++)
		perm[(unIIa + i) % unII] = i;

	return nn_permute_inputs_F(result, unII, perm);
}

/**
 * Duplicate two inputs of a nn_t, i.e. the input b will be set to equal the input a
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... , x_n) = f(x_1, ... ,x_b-1, x_a, x_b+1, ... x_n)
 *
 * The duplicated input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 * Note that this behaviour differs from nlop_dup(operator_dup)
 * Note that the initializers of a and b must be compatible, i.e. at least one is NULL or they equal
 *
 * @param op nn_t struct
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 *
 * @returns nn_t with dupped inputs
 */
nn_t nn_dup(nn_t op, int a, const char* aname, int b, const char* bname)
{

	a = nn_get_in_arg_index(op, a, aname);
	b = nn_get_in_arg_index(op, b, bname);

	unsigned int II = nn_get_nr_in_args(op);
	unsigned int OO = nn_get_nr_out_args(op);

	auto init_tmp = init_dup_create(op->initializers[a], op->initializers[b]);

	auto nlop = nlop_dup(nn_get_nlop(op), MIN(a , b), MAX(a, b));
	if (a > b)
		nlop = nlop_shift_input_F(nlop, a - 1, b);

	auto result = nn_from_nlop_F(nlop);

	for (unsigned int i = 0, ip = 0; i < II - 1; i++) {

		if (i == (unsigned int)b) ip++;
		nn_clone_arg_i_from_i(result, i, op, ip++);
	}
	for (unsigned int i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	initializer_free(result->initializers[(a > b) ? a - 1 : a]);
	result->initializers[(a > b) ? a - 1 : a] = init_tmp;

	return result;
}

/**
 * Duplicate two inputs of a nn_t, i.e. the input b will be set to equal the input a, and free the nn_t
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... , x_n) = f(x_1, ... ,x_b-1, x_a, x_b+1, ... x_n)
 *
 * The duplicated input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 * Note that this behaviour differs from nlop_dup(operator_dup)
 * Note that the initializers of a and b must be compatible, i.e. at least one is NULL or they equal
 *
 * @param op nn_t struct (will be freed)
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 *
 * @returns nn_t with dupped inputs
 */
nn_t nn_dup_F(nn_t op, int a, const char* aname, int b, const char* bname)
{
	auto result = nn_dup(op, a, aname, b, bname);
	nn_free(op);
	return result;
}

/**
 * Stack two inputs of a nn_t, i.e. the new input will be destacked and chained in the inputs a and b
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... x_s ... , x_n) = f(x_1, ... ,x_b-1, x_sb, x_b+1, ..., x_sa, ... x_n)
 * , where x_s equals x_sb stacked on x_sa
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 * @param stack_dim index at which dimension the two inputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked inputs
 */
nn_t nn_stack_inputs(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	a = nn_get_in_arg_index(op, a, aname);
	b = nn_get_in_arg_index(op, b, bname);

	unsigned int II = nn_get_nr_in_args(op);
	unsigned int OO = nn_get_nr_out_args(op);

	auto iova = nlop_generic_domain(op->nlop, a);
	auto iovb = nlop_generic_domain(op->nlop, b);
	assert(iova->N == iovb->N);
	auto init_tmp = init_stack_create(iova->N, stack_dim, iova->dims, op->initializers[a], iovb->dims, op->initializers[b]);

	auto nlop = nlop_stack_inputs(nn_get_nlop(op), a, b, stack_dim);
	if (a > b)
		nlop = nlop_shift_input_F(nlop, a - 1, b);
	auto result = nn_from_nlop_F(nlop);

	for (unsigned int i = 0, ip = 0; i < II - 1; i++) {

		if (i == (unsigned int)b) ip++;
		nn_clone_arg_i_from_i(result, i, op, ip++);
	}
	for (unsigned int i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	initializer_free(result->initializers[(a > b) ? a - 1 : a]);
	result->initializers[(a > b) ? a - 1 : a] = init_tmp;

	return result;
}

/**
 * Stack two inputs of a nn_t, i.e. the new input will be destacked and chained in the inputs a and b, and frees the nn_t
 *
 * g(x_1, ... ,x_b-1, x_b+1, ... x_s ... , x_n) = f(x_1, ... ,x_b-1, x_sb, x_b+1, ..., x_sa, ... x_n)
 * , where x_s equals x_sb stacked on x_sa
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct (will be freed)
 * @param a first input index (ignored if aname != NULL)
 * @param aname name of first input
 * @param b second input index (ignored if bname != NULL)
 * @param bname name of second input
 * @param stack_dim index at which dimension the two inputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked inputs
 */
nn_t nn_stack_inputs_F(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	auto result = nn_stack_inputs(op, a, aname, b, bname, stack_dim);
	nn_free(op);
	return result;
}

/**
 * Stack two outputs of a nn_t, i.e. the two outputs a and b will be computed and the results of b is stacked on the result of a to form the new output
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct
 * @param a first output index (ignored if aname != NULL)
 * @param aname name of first output
 * @param b second output index (ignored if bname != NULL)
 * @param bname name of second output
 * @param stack_dim index at which dimension the two outputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked outputs
 */
nn_t nn_stack_outputs(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	a = nn_get_out_arg_index(op, a, aname);
	b = nn_get_out_arg_index(op, b, bname);

	unsigned int II = nn_get_nr_in_args(op);
	unsigned int OO = nn_get_nr_out_args(op);

	auto nlop = nlop_stack_outputs(nn_get_nlop(op), a, b, stack_dim);
	if (a > b)
		nlop = nlop_shift_output_F(nlop, a - 1, b);
	auto result = nn_from_nlop_F(nlop);

	for (unsigned int i = 0; i < II; i++)
		nn_clone_arg_i_from_i(result, i, op, i);

	for (unsigned int i = 0, ip = 0; i < OO - 1; i++) {

		if (i == (unsigned int)b) ip++;
		nn_clone_arg_o_from_o(result, i, op, ip++);
	}

	return result;
}

/**
 * Stack two outputs of a nn_t, i.e. the two outputs a and b will be computed and the results of b is stacked on the result of a to form the new output, and free nn_t
 *
 * The stacked input will take the position, name, etc. of the index a/aname.
 * Thus, if a and b are numeric indices and a>b, the duplicated index will be a-1, and a if a<b.
 *
 * @param op nn_t struct (will be freed)
 * @param a first output index (ignored if aname != NULL)
 * @param aname name of first output
 * @param b second output index (ignored if bname != NULL)
 * @param bname name of second output
 * @param stack_dim index at which dimension the two outputs should be stacked (can be negative meaning that it is counted backward starting from the last)
 *
 * @returns nn_t with stacked outputs
 */
nn_t nn_stack_outputs_F(nn_t op, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	auto result = nn_stack_outputs(op, a, aname, b, bname, stack_dim);
	nn_free(op);
	return result;
}

/**
 * Permute inputs of nn_t such that the index o is shifted to position n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted inputs
 *
 * @note the indices n and o also count named arguments, use nn_shift_input_F if only unnamed inputs shall be shifted
 */
nn_t nn_shift_input_index_F(nn_t x, uint n, uint o)
{
	int new_index = n;
	int old_index = o;
	int II = nn_get_nr_in_args(x);
	assert(old_index < II);
	assert(new_index < II);

	int perm[II];
	for (int i = 0, ip = 0; i < II; i++, ip++) {

		perm[i] = ip;
		if (i == old_index) ip++;
		if (i == new_index) ip--;
		if (new_index > old_index)
			perm[i] = ip;
	}

	perm[new_index] = old_index;

	return nn_permute_inputs_F(x, II, perm);
}

/**
 * Permute outputs of nn_t such that the index o is shifted to position n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted outputs
 *
 * @note the indices n and o also count named arguments, use nn_shift_output_F if only unnamed inputs shall be shifted
 */
nn_t nn_shift_output_index_F(nn_t x, uint n, uint o)
{
	int new_index = n;
	int old_index = o;
	int OO = nn_get_nr_out_args(x);
	assert(old_index < OO);
	assert(new_index < OO);

	int perm[OO];
	for (int i = 0, ip = 0; i < OO; i++, ip++) {

		perm[i] = ip;
		if (i == old_index) ip++;
		if (i == new_index) ip--;
		if (new_index > old_index)
			perm[i] = ip;
	}

	perm[new_index] = old_index;

	return nn_permute_outputs_F(x, OO, perm);
}

/**
 * Permute inputs of nn_t such that the input at o is shifted to position of input n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted inputs
 */
nn_t nn_shift_input_F(nn_t x, int n, const char* nname, int o, const char* oname)
{
	n = nn_get_in_arg_index(x, n, nname);
	o = nn_get_in_arg_index(x, o, oname);

	return nn_shift_input_index_F(x, n, o);
}

/**
 * Permute outputs of nn_t such that the output o is shifted to position of output n and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param n new position of input
 * @param o old position of input
 *
 * @returns nn_t with permuted outputs
 */
nn_t nn_shift_output_F(nn_t x, int n, const char* nname, int o, const char* oname)
{
	n = nn_get_out_arg_index(x, n, nname);
	o = nn_get_out_arg_index(x, o, oname);

	return nn_shift_output_index_F(x, n, o);
}

/**
 * Rename input with name to #DUP_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to dup the two inputs
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_dup_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 6];
	sprintf(nname, "#DUP_%s", name);

	return nn_rename_input_F(x, nname, name);
}

/**
 * Rename input with name to #STACK_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to stack the two inputs
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_stack_input_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 8];
	sprintf(nname, "#STACK_%s", name);

	return nn_rename_input_F(x, nname, name);
}

/**
 * Rename output with name to #STACK_name and free nn_t
 *
 * If the nn_t is combined with another having an input with the selected name,
 * nn_stack_dup_by_name_F can be used to stack the two outputs
 *
 * @param op nn_t struct (will be freed)
 * @param name old name
 *
 * @returns nn_t with new name
 */
nn_t nn_mark_stack_output_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 8];
	sprintf(nname, "#STACK_%s", name);

	return nn_rename_output_F(x, nname, name);
}

static nn_t stack_in_by_name(nn_t x) {

	const char* stack_name = NULL;
	for (unsigned int i = 0; i < nn_get_nr_in_args(x); i ++)
		if (NULL != x->in_names[i] && 0 == strncmp(x->in_names[i], "#STACK_", 7))
			stack_name = x->in_names[i];

	if (NULL == stack_name)
		return x;

	return nn_stack_inputs_F(x, 0, stack_name + 7, 0, stack_name, -1);
}

static nn_t stack_out_by_name(nn_t x) {

	const char* stack_name = NULL;
	for (unsigned int i = 0; i < nn_get_nr_out_args(x); i ++)
		if (NULL != x->out_names[i] && 0 == strncmp(x->out_names[i], "#STACK_", 7))
			stack_name = x->out_names[i];

	if (NULL == stack_name)
		return x;

	return nn_stack_outputs_F(x, 0, stack_name + 7, 0, stack_name, -1);
}

static nn_t dup_by_name(nn_t x) {

	const char* stack_name = NULL;
	for (unsigned int i = 0; i < nn_get_nr_in_args(x); i ++)
		if (NULL != x->in_names[i] && 0 == strncmp(x->in_names[i], "#DUP_", 5))
			stack_name = x->in_names[i];

	if (NULL == stack_name)
		return x;

	return nn_dup_F(x, 0, stack_name + 5, 0, stack_name);
}

/**
 * Search for input/oputput names #DUP_%s or #STACK_%s. If such names are found, a nn_t with stacked (#STACK_%s on %s) and dupped inputs/outputs is returned
 *
 * @param op nn_t struct (will be freed)
 *
 * @returns nn_t with stacked and dupped arguments
 */
nn_t nn_stack_dup_by_name_F(nn_t op)
{
	nn_t result = op;

	nn_t prev = NULL;
	while (result != prev) {

		prev = result;
		result = dup_by_name(result);
	}

	prev = NULL;
	while (result != prev) {

		prev = result;
		result = stack_in_by_name(result);
	}

	prev = NULL;
	while (result != prev) {

		prev = result;
		result = stack_out_by_name(result);
	}

	return result;
}

static bool is_name_in_list(int N, const char* names[N], const char* name)
{
	if (NULL == name)
		return false;

	bool result = false;
	for (int i = 0; i < N; i++)
		result |= (NULL == names[i]) ? false : (0 == strcmp(names[i], name));
	return result;
}

/**
 * Permute inputs of nn_t such that all inputs with a name contained in the provided list are in the same order as in the list and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param N no. of names in the list
 * @param sorted_names list of names
 *
 * @returns nn_t with sorted inputs
 *
 * @note not all input names must be provided in the list and vice versa
 */
nn_t nn_sort_inputs_by_list_F(nn_t x, unsigned int N, const char* sorted_names[N])
{
	unsigned int II = nn_get_nr_in_args(x);
	int nperm[II];

	int index = 0;

	for (uint i = 0; i < II; i++){

		if (is_name_in_list(N, sorted_names, nn_get_in_name_from_arg_index(x, i))) {

			while (! nn_is_name_in_in_args(x, sorted_names[index]))
				index++;

			nperm[i] = nn_get_in_arg_index(x, 0, sorted_names[index]);
			index++;
		} else {

			nperm[i] = i;
		}
	}

	return nn_permute_inputs_F(x, II, nperm);
}

/**
 * Permute inputs of nn_t such that all inputs with a name contained in the provided list are in the same order as in the list and free nn_t
 *
 * @param op nn_t struct (will be freed)
 * @param N no. of names in the list
 * @param sorted_names list of names
 *
 * @returns nn_t with sorted inputs
 *
 * @note not all input names must be provided in the list and vice versa
 */
nn_t nn_sort_outputs_by_list_F(nn_t x, unsigned int N, const char* sorted_names[N])
{
	unsigned int OO = nn_get_nr_out_args(x);
	int nperm[OO];

	int index = 0;

	for (uint i = 0; i < OO; i++){

		if (is_name_in_list(N, sorted_names, nn_get_out_name_from_arg_index(x, i))) {

			while (! nn_is_name_in_out_args(x, sorted_names[index]))
				index++;

			nperm[i] = nn_get_out_arg_index(x, 0, sorted_names[index]);
			index++;
		} else {

			nperm[i] = i;
		}
	}

	return nn_permute_outputs_F(x, OO, nperm);
}
