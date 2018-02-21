
#include <stddef.h>
#include <assert.h>

#include "num/ops.h"
#include "num/iovec.h"

#include "misc/misc.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"

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




struct nlop_s* nlop_combine(const struct nlop_s* a, const struct nlop_s* b)
{
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

				//assert(dom->N == cod->N);
				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(dom->N,
					cod->dims, cod->strs, dom->dims, dom->strs);

			} else
			if ((ai <= i) && (o < ao)) {

				auto dom = nlop_generic_domain(b, i - ai);
				auto cod = nlop_generic_codomain(a, o);

				assert(dom->N == cod->N);
				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(dom->N,
					cod->dims, cod->strs, dom->dims, dom->strs);
			}
		}
	}


	n->op = operator_combi_create(2, (const struct operator_s*[]){
			operator_ref(a->op), operator_ref(b->op) });

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

	n->op = operator_permute(n->op, II + OO, perm);

	return PTR_PASS(n);
}


