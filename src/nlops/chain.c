
#include <stddef.h>

#include "num/ops.h"

#include "misc/misc.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"

#include "linops/linop.h"

#include "chain.h"




struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b)
{
#if 0
	const struct linop_s* la = linop_from_nlop(a);
	const struct linop_s* lb = linop_from_nlop(b);

	if ((NULL != la) && (NULL != lb))
		return nlop_from_linop(linop_chain(la, lb));
#endif
	PTR_ALLOC(struct nlop_s, n);
#if 0
	if (NULL == la)
		la = a->derivative;

	if (NULL == lb)
		lb = b->derivative;
#endif
	n->op = operator_chain(a->op, b->op);
#if 0
	n->derivative = linop_chain(la, lb);
#else
	n->derivative = linop_chain(a->derivative, b->derivative);
#endif
	return PTR_PASS(n);
}




