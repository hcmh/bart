
#include <stddef.h>

#include "num/ops.h"

#include "misc/misc.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"

#include "linops/linop.h"

#include "chain.h"




struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b)
{
	const struct linop_s* la = linop_from_nlop(a);
	const struct linop_s* lb = linop_from_nlop(b);

	if ((NULL != a) && (NULL != b))
		return nlop_from_linop(linop_chain(la, lb));

	PTR_ALLOC(struct nlop_s, n);

	if (NULL == la)
		la = a->derivative;

	if (NULL == lb)
		lb = b->derivative;

	n->op = operator_chain(a->op, b->op);
	n->derivative = linop_chain(la, lb);

	return PTR_PASS(n);
}




