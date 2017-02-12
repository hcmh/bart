
#include <assert.h>

#include "misc/misc.h"
#include "misc/types.h"

#include "num/iovec.h"
#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "nn.h"


struct relu_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* domain;
	const struct iovec_s* codomain;
};

DEF_TYPEID(relu_s);

static void relu_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
        const struct relu_s* d = CAST_DOWN(relu_s, _data);
	assert(2 == N);
	// FIXMLE
        md_smax2(d->domain->N, d->domain->dims, d->codomain->strs, args[0], d->domain->strs, args[1], 0.);
}


static void relu_free(const operator_data_t* _data)
{
        const struct relu_s* d = CAST_DOWN(relu_s, _data);
	iovec_free(d->domain);
	iovec_free(d->codomain);
	xfree(d);
}


const struct operator_s* operator_relu_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N])
{
	PTR_ALLOC(struct relu_s, data);
	SET_TYPEID(relu_s, data);

        data->domain = iovec_create2(N, dims, istrs, CFL_SIZE);
        data->codomain = iovec_create2(N, dims, ostrs, CFL_SIZE);

        return operator_create2(N, dims, ostrs, N, dims, istrs, CAST_UP(PTR_PASS(data)), relu_apply, relu_free);
}

const struct operator_s* operator_relu_create(unsigned int N, const long dims[N])
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
        return operator_relu_create2(N, dims, strs, strs);
}


