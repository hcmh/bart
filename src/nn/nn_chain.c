#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "misc/misc.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "nn/nn.h"
#include "nn_chain.h"


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
nn_t nn_reshape_in(nn_t op, int i, const char* iname, int N, const long idims[N])
{
	i = nn_get_in_arg_index(op, i, iname);
	auto result = nn_from_nlop_F(nlop_reshape_in(nn_get_nlop(op), i, N, idims));

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (unsigned int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	return result;
}
nn_t nn_reshape_out_F(nn_t op, int o, const char* oname, int NO, const long odims[NO])
{
	auto result = nn_reshape_out(op, o, oname, NO, odims);
	nn_free(op);
	return result;
}
nn_t nn_reshape_in_F(nn_t op, int i, const char* iname, int NI, const long idims[NI])
{
	auto result = nn_reshape_in(op, i, iname, NI, idims);
	nn_free(op);
	return result;
}

nn_t nn_append_singleton_dim_in_F(nn_t op, int i, const char* iname)
{
	auto iov = nn_generic_domain(op, i, iname);
	long dims[iov->N + 1];
	md_copy_dims(iov->N, dims, iov->dims);
	dims[iov->N] = 1;

	return nn_reshape_in_F(op, i, iname, iov->N + 1, dims);
}

nn_t nn_append_singleton_dim_out_F(nn_t op, int o, const char* oname)
{
	auto iov = nn_generic_codomain(op, o, oname);
	long dims[iov->N + 1];
	md_copy_dims(iov->N, dims, iov->dims);
	dims[iov->N] = 1;

	return nn_reshape_out_F(op, o, oname, iov->N + 1, dims);
}


nn_t nn_permute_inputs(nn_t x, unsigned int I2, const int perm[I2])
{
	assert((nn_get_nr_in_args(x) == I2) || (nn_get_nr_unnamed_in_args(x) == I2));

	unsigned int II = nn_get_nr_in_args(x);
	int nperm[II];

	for (unsigned int i = 0; i < II; i++)
		nperm[i] = i;

	if (nn_get_nr_unnamed_in_args(x) == I2) {

		for(unsigned int i = 0; i < nn_get_nr_unnamed_in_args(x); i++)
			nperm[nn_get_in_arg_index(x, i, NULL)] = nn_get_in_arg_index(x, perm[i], NULL);
	} else {

		for(unsigned int i = 0; i < nn_get_nr_in_args(x); i++)
			nperm[i] = perm[i];
	}

	auto result = nn_from_nlop_F(nlop_permute_inputs(nn_get_nlop(x), II, nperm));

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, x, nperm[i]);
	for (unsigned int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, x, i);

	return result;
}

nn_t nn_permute_outputs(nn_t x, unsigned int O2, const int perm[O2])
{
	assert((nn_get_nr_out_args(x) == O2) || (nn_get_nr_unnamed_out_args(x) == O2));

	unsigned int OO = nn_get_nr_out_args(x);
	int nperm[OO];

	for (unsigned int i = 0; i < OO; i++)
		nperm[i] = i;

	if (nn_get_nr_unnamed_out_args(x) == O2) {

		for(unsigned int i = 0; i < nn_get_nr_unnamed_out_args(x); i++)
			nperm[nn_get_out_arg_index(x, i, NULL)] = nn_get_out_arg_index(x, perm[i], NULL);
	} else {

		for(unsigned int i = 0; i < nn_get_nr_out_args(x); i++)
			nperm[i] = perm[i];
	}

	auto result = nn_from_nlop_F(nlop_permute_outputs(nn_get_nlop(x), OO, nperm));

	for (unsigned int i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, x, i);
	for (unsigned int i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, x, nperm[i]);

	return result;
}


nn_t nn_permute_inputs_F(nn_t x, unsigned int I2, const int perm[I2])
{
	auto result = nn_permute_inputs(x, I2, perm);
	nn_free(x);
	return result;
}
nn_t nn_permute_outputs_F(nn_t x, unsigned int O2, const int perm[O2])
{
	auto result = nn_permute_outputs(x, O2, perm);
	nn_free(x);
	return result;
}
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

nn_t nn_combine_FF(nn_t a, nn_t b)
{
	auto result = nn_combine(a, b);
	nn_free(a);
	nn_free(b);
	return result;
}

nn_t nn_link(nn_t x, int o, const char* oname, int i, const char* iname)
{
	unsigned int OO = nn_get_nr_out_args(x) - 1;
	unsigned int II = nn_get_nr_in_args(x) - 1;

	o = nn_get_out_arg_index(x, o, oname);
	i = nn_get_in_arg_index(x, i, iname);

	auto result = nn_from_nlop_F(nlop_link(nn_get_nlop(x), o, i));

	for (unsigned int ii = 0, ip = 0; ii < II; ii++){

		if ((int)ii == i)
			ip++;
		nn_clone_arg_i_from_i(result, ii, x, ip);
		ip++;
	}

	for (unsigned int oo = 0, op = 0; oo < OO; oo++){

		if ((int)oo == o)
			op++;
		nn_clone_arg_o_from_o(result, oo, x, op);
		op++;
	}

	return result;
}
nn_t nn_link_F(nn_t x, int o, const char* oname, int i, const char* iname)
{
	auto result = nn_link(x, o, oname, i, iname);
	nn_free(x);
	return result;
}

nn_t nn_chain2(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int OO = nn_get_nr_unnamed_out_args(b);
	return nn_link_F(nn_combine(b, a), (NULL == oname) ? o + OO : 0, oname, i, iname);
}
nn_t nn_chain2_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname)
{
	int OO = nn_get_nr_unnamed_out_args(b);
	return nn_link_F(nn_combine_FF(b, a), (NULL == oname) ? o + OO : 0, oname, i, iname);
}
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

nn_t nn_dup(nn_t x, int a, const char* aname, int b, const char* bname)
{

	a = nn_get_in_arg_index(x, a, aname);
	b = nn_get_in_arg_index(x, b, bname);

	unsigned int II = nn_get_nr_in_args(x);
	unsigned int OO = nn_get_nr_out_args(x);

	auto nlop = nlop_dup(nn_get_nlop(x), MIN(a , b), MAX(a, b));
	if (a > b)
		nlop = nlop_shift_input_F(nlop, a - 1, b);

	auto result = nn_from_nlop_F(nlop);

	for (unsigned int i = 0, ip = 0; i < II - 1; i++) {

		if (i == (unsigned int)b) ip++;
		nn_clone_arg_i_from_i(result, i, x, ip++);
	}
	for (unsigned int i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, x, i);

	return result;
}

nn_t nn_dup_F(nn_t x, int a, const char* aname, int b, const char* bname)
{
	auto result = nn_dup(x, a, aname, b, bname);
	nn_free(x);
	return result;
}

nn_t nn_stack_inputs(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	a = nn_get_in_arg_index(x, a, aname);
	b = nn_get_in_arg_index(x, b, bname);

	unsigned int II = nn_get_nr_in_args(x);
	unsigned int OO = nn_get_nr_out_args(x);

	auto nlop = nlop_stack_inputs(nn_get_nlop(x), a, b, stack_dim);
	if (a > b)
		nlop = nlop_shift_input_F(nlop, a - 1, b);
	auto result = nn_from_nlop_F(nlop);

	for (unsigned int i = 0, ip = 0; i < II - 1; i++) {

		if (i == (unsigned int)b) ip++;
		nn_clone_arg_i_from_i(result, i, x, ip++);
	}
	for (unsigned int i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, x, i);
	return result;
}
nn_t nn_stack_inputs_F(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	auto result = nn_stack_inputs(x, a, aname, b, bname, stack_dim);
	nn_free(x);
	return result;
}
nn_t nn_stack_outputs(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	a = nn_get_out_arg_index(x, a, aname);
	b = nn_get_out_arg_index(x, b, bname);

	unsigned int II = nn_get_nr_in_args(x);
	unsigned int OO = nn_get_nr_out_args(x);

	auto nlop = nlop_stack_outputs(nn_get_nlop(x), a, b, stack_dim);
	if (a > b)
		nlop = nlop_shift_output_F(nlop, a - 1, b);
	auto result = nn_from_nlop_F(nlop);

	for (unsigned int i = 0; i < II; i++)
		nn_clone_arg_i_from_i(result, i, x, i);

	for (unsigned int i = 0, ip = 0; i < OO - 1; i++) {

		if (i == (unsigned int)b) ip++;
		nn_clone_arg_o_from_o(result, i, x, ip++);
	}

	return result;
}

nn_t nn_stack_outputs_F(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim)
{
	auto result = nn_stack_outputs(x, a, aname, b, bname, stack_dim);
	nn_free(x);
	return result;
}


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

nn_t nn_shift_input_F(nn_t x, int n, const char* nname, int o, const char* oname)
{
	n = nn_get_in_arg_index(x, n, nname);
	o = nn_get_in_arg_index(x, o, oname);

	return nn_shift_input_index_F(x, n, o);
}

nn_t nn_shift_output_F(nn_t x, int n, const char* nname, int o, const char* oname)
{
	n = nn_get_out_arg_index(x, n, nname);
	o = nn_get_out_arg_index(x, o, oname);

	return nn_shift_output_index_F(x, n, o);
}

nn_t nn_mark_dup_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 6];
	sprintf(nname, "#DUP_%s", name);

	return nn_rename_input_F(x, nname, name);
}

nn_t nn_mark_stack_input_F(nn_t x, const char* name)
{
	char nname[strlen(name) + 8];
	sprintf(nname, "#STACK_%s", name);

	return nn_rename_input_F(x, nname, name);
}

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

nn_t nn_stack_dup_by_name_F(nn_t x)
{

	nn_t result = x;

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
