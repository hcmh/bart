/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */



#include <stddef.h>
#include <assert.h>

#include "misc/debug.h"

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "misc/misc.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/stack.h"

#include "linops/linop.h"

#include "chain.h"




struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b)
{
	assert(1 == nlop_get_nr_in_args(a));
	assert(1 == nlop_get_nr_out_args(a));
	assert(1 == nlop_get_nr_in_args(b));
	assert(1 == nlop_get_nr_out_args(b));

	const struct linop_s* la = linop_from_nlop(a);
	const struct linop_s* lb = linop_from_nlop(b);

	if ((NULL != la) && (NULL != lb))
		return nlop_from_linop(linop_chain(la, lb));

	PTR_ALLOC(struct nlop_s, n);

	const struct linop_s* (*der)[1][1] = TYPE_ALLOC(const struct linop_s*[1][1]);
	n->derivative = &(*der)[0][0];

	if (NULL == la)
		la = a->derivative[0];

	if (NULL == lb)
		lb = b->derivative[0];

	n->op = operator_chain(a->op, b->op);
	n->derivative[0] = linop_chain(la, lb);

	return PTR_PASS(n);
}

struct nlop_s* nlop_chain_FF(const struct nlop_s* a, const struct nlop_s* b)
{
	struct nlop_s* x = nlop_chain(a, b);
	nlop_free(a);
	nlop_free(b);
	return x;
}


struct nlop_s* nlop_chain2(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
//	int ai = nlop_get_nr_in_args(a);
//	int ao = nlop_get_nr_out_args(a);
//	int bi = nlop_get_nr_in_args(b);
	int bo = nlop_get_nr_out_args(b);
#if 0
	if ((1 == ai) && (1 == ao) && (1 == bi) && (1 == bo)) {

		assert((0 == o) && (0 == i));
		return nlop_chain(a, b);
	}
#endif

	struct nlop_s* nl = nlop_combine(b, a);
	struct nlop_s* li = nlop_link(nl, bo + o, i);
	nlop_free(nl);

	return li;
}

struct nlop_s* nlop_chain2_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
	auto result = nlop_chain2(a, o, b, i);

	nlop_free(a);
	nlop_free(b);

	return result;
}

/*
 * Chains two non-linear operators
 * permutes inputs to have order: inputs a, inputs b
 */
struct nlop_s* nlop_chain2_swap_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
	auto result = nlop_chain2(a, o, b, i);
	int permute_array[nlop_get_nr_in_args(result)];
	for (int i = 0; i < nlop_get_nr_in_args(result); i++)
		permute_array[(nlop_get_nr_in_args(a) + i) % nlop_get_nr_in_args(result)] = i;

	result = nlop_permute_inputs_F(result, nlop_get_nr_in_args(result), permute_array);

	nlop_free(a);
	nlop_free(b);

	return result;
}

/*
 * CAVE: if we pass the same operator twice, it might not
 * as they store state with respect to the derivative
 */
struct nlop_s* nlop_combine(const struct nlop_s* a, const struct nlop_s* b)
{
	assert(a != b);	// could also be deeply nested, but we do not detect it

	int ai = nlop_get_nr_in_args(a);
	int ao = nlop_get_nr_out_args(a);
	int bi = nlop_get_nr_in_args(b);
	int bo = nlop_get_nr_out_args(b);

	PTR_ALLOC(struct nlop_s, n);

	int II = ai + bi;
	int OO = ao + bo;

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++) {
		for (int o = 0; o < OO; o++) {

			if ((i < ai) && (o < ao))
				(*der)[i][o] = linop_clone(nlop_get_derivative(a, o, i));
			else
			if ((ai <= i) && (ao <= o))
				(*der)[i][o] = linop_clone(nlop_get_derivative(b, o - ao, i - ai));
			else
			if ((i < ai) && (ao <= o)) {

				auto dom = nlop_generic_domain(a, i);
				auto cod = nlop_generic_codomain(b, o - ao);

				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(cod->N,
					cod->dims, cod->strs, dom->N, dom->dims, dom->strs);

			} else
			if ((ai <= i) && (o < ao)) {

				auto dom = nlop_generic_domain(b, i - ai);
				auto cod = nlop_generic_codomain(a, o);

				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(cod->N,
					cod->dims, cod->strs, dom->N, dom->dims, dom->strs);
			}
		}
	}


	auto cop = operator_combi_create(2, (const struct operator_s*[]){ a->op, b->op });

	assert(II == (int)operator_nr_in_args(cop));
	assert(OO == (int)operator_nr_out_args(cop));

	int perm[II + OO];	// ao ai bo bi -> ao bo ai bi
	int p = 0;

	for (int i = 0; i < ao; i++)
		perm[p++] = i;

	for (int i = 0; i < bo; i++)
		perm[p++] = (ao + ai + i);

	for (int i = 0; i < ai; i++)
		perm[p++] = (ao + i);

	for (int i = 0; i < bi; i++)
		perm[p++] = (ao + ai + bo + i);

	assert(II + OO == p);

	n->op = operator_permute(cop, II + OO, perm);
	operator_free(cop);

	return PTR_PASS(n);
}

struct nlop_s* nlop_combine_FF(const struct nlop_s* a, const struct nlop_s* b)
{
	auto result = nlop_combine(a, b);
	nlop_free(a);
	nlop_free(b);
	return result;
}



struct nlop_s* nlop_link(const struct nlop_s* x, int oo, int ii)
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(ii < II);
	assert(oo < OO);

	PTR_ALLOC(struct nlop_s, n);
	PTR_ALLOC(const struct linop_s*[II - 1][OO - 1], der);

	assert(operator_ioflags(x->op) == ((1u << OO) - 1));

	n->op = operator_link_create(x->op, oo, OO + ii);

	assert(operator_ioflags(n->op) == ((1u << (OO - 1)) - 1));

	// f(x_1, ..., g(x_n+1, ..., x_n+m), ..., xn)

	for (int i = 0, ip = 0; i < II - 1; i++, ip++) {

		if (i == ii)
			ip++;

		for (int o = 0, op = 0; o < OO - 1; o++, op++) {

			if (o == oo)
				op++;

			const struct linop_s* tmp = linop_chain(nlop_get_derivative(x, oo, ip),
								nlop_get_derivative(x, op, ii));

			(*der)[i][o] = linop_plus(
				nlop_get_derivative(x, op, ip),
				tmp);

			linop_free(tmp);
		}
	}

	n->derivative = &(*PTR_PASS(der))[0][0];

	return PTR_PASS(n);
}

struct nlop_s* nlop_link_F(const struct nlop_s* x, int oo, int ii)
{
	auto result = nlop_link(x, oo, ii);
	nlop_free(x);
	return result;
}


struct nlop_s* nlop_dup(const struct nlop_s* x, int a, int b)
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(a < II);
	assert(b < II);
        assert(a < b);

	PTR_ALLOC(struct nlop_s, n);
	PTR_ALLOC(const struct linop_s*[II-1][OO], der);

	assert(operator_ioflags(x->op) == ((1u << OO) - 1));

	n->op = operator_dup_create(x->op, OO + a, OO + b);

	assert(operator_ioflags(n->op) == ((1u << OO) - 1));

	// f(x_1, ..., xa, ... xa, ..., xn)

	for (int i = 0, ip = 0; i < II - 1; i++, ip++) {

		if (i == b)
			ip++;

		for (int o = 0; o < OO; o++) {

                        if (i == a)
				(*der)[i][o] = linop_plus(nlop_get_derivative(x, o, ip), nlop_get_derivative(x, o, b));
			else
				(*der)[i][o] = linop_clone(nlop_get_derivative(x, o, ip));

		}
	}

	n->derivative = &(*PTR_PASS(der))[0][0];

	return PTR_PASS(n);
}

struct nlop_s* nlop_stack_inputs(const struct nlop_s* x, int a, int b, unsigned long stack_dim)
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(a < II);
	assert(b < II);
	assert( a!= b);

	auto doma = nlop_generic_domain(x, a);
	auto domb = nlop_generic_domain(x, b);
	assert(doma->N == domb->N);

	long N = doma->N;
	long idims[N];
	md_copy_dims(N, idims, doma->dims);
	idims[stack_dim] += domb->dims[stack_dim];
	auto nlop_destack = nlop_destack_create(N, doma->dims, domb->dims, idims, stack_dim);
	auto combined = nlop_combine(x, nlop_destack);
	auto result = nlop_link_F(combined, OO + 1, b);
	result = nlop_link_F(result, OO, a < b ? a : a - 1);
	nlop_free(nlop_destack);

	int perm[II-1];
	for (int i = 0; i < II - 1; i++)
		perm[i] = (i <= MIN(a, b)) ? i : i - 1;
	perm[MIN(a, b)] = II - 2;

	return nlop_permute_inputs_F(result, II - 1, perm);
}

struct nlop_s* nlop_stack_inputs_F(const struct nlop_s* x, int a, int b, unsigned long stack_dim)
{
	auto result = nlop_stack_inputs(x, a, b, stack_dim);
	nlop_free(x);
	return result;
}

//renamed for consistency, deprecated
struct nlop_s* nlop_destack(const struct nlop_s* x, int a, int b, unsigned long stack_dim)
{
	return nlop_stack_inputs(x, a, b, stack_dim);
}

//renamed for consistency, deprecated
struct nlop_s* nlop_destack_F(const struct nlop_s* x, int a, int b, unsigned long stack_dim)
{
	return nlop_stack_inputs_F(x, a, b, stack_dim);
}




struct nlop_s* nlop_stack_outputs(const struct nlop_s* x, int a, int b, unsigned long stack_dim)
{
	//int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(a < OO);
	assert(b < OO);
	assert(a != b);

	auto codoma = nlop_generic_codomain(x, a);
	auto codomb = nlop_generic_codomain(x, b);
	assert(codoma->N == codomb->N);

	long N = codoma->N;
	long odims[N];
	md_copy_dims(N, odims, codoma->dims);
	odims[stack_dim] += codomb->dims[stack_dim];
	auto nlop_stack = nlop_stack_create(N, odims, codoma->dims, codomb->dims, stack_dim);
	auto combined = nlop_combine(nlop_stack, x);
	nlop_free(nlop_stack);

	auto result = nlop_link_F(combined, b + 1, 1);
	result = nlop_link_F(result, a < b ? a + 1 : a, 0);

	int perm[OO - 1];
	for (int o = 0; o < OO - 1 ; o++)
		perm[o] = (o <= MIN(a, b)) ? o + 1 : o;
	perm[MIN(a, b)] = 0;

	return nlop_permute_outputs_F(result, OO - 1, perm);
}

struct nlop_s* nlop_stack_outputs_F(const struct nlop_s* x, int a, int b, unsigned long stack_dim)
{
	auto result = nlop_stack_outputs(x, a, b, stack_dim);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_dup_F(const struct nlop_s* x, int a, int b)
{
	auto result = nlop_dup(x, a, b);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_permute_inputs(const struct nlop_s* x, int I2, const int perm[I2])
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(II == I2);

	PTR_ALLOC(struct nlop_s, n);

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*der)[i][o] = linop_clone(nlop_get_derivative(x, o, perm[i]));

	int perm2[II + OO];

	for (int i = 0; i < II + OO; i++)
		perm2[i] = (i < OO) ? i : (OO + perm[i - OO]);

	n->op = operator_permute(x->op, II + OO, perm2);

	return PTR_PASS(n);
}

struct nlop_s* nlop_permute_inputs_F(const struct nlop_s* x, int I2, const int perm[I2])
{
	auto result = nlop_permute_inputs(x, I2, perm);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_permute_outputs(const struct nlop_s* x, int O2, const int perm[O2])
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(OO == O2);

	PTR_ALLOC(struct nlop_s, n);

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*der)[i][o] = linop_clone(nlop_get_derivative(x, perm[o], i));


	int perm2[II + OO];

	for (int i = 0; i < II + OO; i++)
		perm2[i] = ((i < OO) ? perm[i] : i);

	n->op = operator_permute(x->op, II + OO, perm2);

	return PTR_PASS(n);
}

struct nlop_s* nlop_permute_outputs_F(const struct nlop_s* x, int O2, const int perm[O2])
{
	auto result = nlop_permute_outputs(x, O2, perm);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_shift_input(const struct nlop_s* x, int new_index, unsigned int old_index)
{
	int II = nlop_get_nr_in_args(x);
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

	return nlop_permute_inputs(x, II, perm);
}

struct nlop_s* nlop_shift_input_F(const struct nlop_s* x, int new_index, unsigned int old_index)
{
	auto result = nlop_shift_input(x, new_index, old_index);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_shift_output(const struct nlop_s* x, int new_index, unsigned int old_index)
{
	int OO = nlop_get_nr_out_args(x);
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

	return nlop_permute_outputs(x, OO, perm);
}

struct nlop_s* nlop_shift_output_F(const struct nlop_s* x, int new_index, unsigned int old_index)
{
	auto result = nlop_shift_output(x, new_index, old_index);
	nlop_free(x);
	return result;
}
