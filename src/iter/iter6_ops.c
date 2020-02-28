#include <assert.h>

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"

#include "iter/iter6.h"
#include "iter6_ops.h"




struct adadelta_update_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* dom;
	float* floating_dx;
	float* floating_g;

	float rho;
	float lr;
	float epsilon;
};

static DEF_TYPEID(adadelta_update_s);

static void adadelta_update_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	struct adadelta_update_s* d = CAST_DOWN(adadelta_update_s, _data);
	assert(2 == N);

	if ((NULL == d->floating_g) || (NULL == d->floating_dx)){

		d->floating_g = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, args[0]);
		d->floating_dx = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, args[0]);
		md_clear(d->dom->N, d->dom->dims, d->floating_g, d->dom->size);
		md_clear(d->dom->N, d->dom->dims, d->floating_dx, d->dom->size);
	}

	float* dst = (float*)args[0];
	float* src = (float*)args[1];

	//Accumulate E[g²]
	md_smul(d->dom->N, d->dom->dims, d->floating_g, d->floating_g, d->rho / (1. - d->rho));
	md_fmac(d->dom->N, d->dom->dims, d->floating_g, src, src);
	md_smul(d->dom->N, d->dom->dims, d->floating_g, d->floating_g, 1. - d->rho);

	//Compute RMS[g]
	float* rmsg = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, args[0]);
	md_sadd(d->dom->N, d->dom->dims, rmsg, d->floating_g, d->epsilon);
	md_sqrt(d->dom->N, d->dom->dims, rmsg, rmsg);

	//Compute RMS[x]/RMS[g]
	float* rmsx_rmsg = md_alloc_sameplace(d->dom->N, d->dom->dims, d->dom->size, args[0]);
	md_sadd(d->dom->N, d->dom->dims, rmsx_rmsg, d->floating_dx, d->epsilon);
	md_sqrt(d->dom->N, d->dom->dims, rmsx_rmsg, rmsx_rmsg);
	md_div(d->dom->N, d->dom->dims, rmsx_rmsg, rmsx_rmsg, rmsg);
	md_free(rmsg);

	//Compute dx
	md_clear(d->dom->N, d->dom->dims, dst, d->dom->size);
	md_fmac(d->dom->N, d->dom->dims, dst, rmsx_rmsg, src);
	md_smul(d->dom->N, d->dom->dims, dst, dst, -d->lr);
	md_free(rmsx_rmsg);

	//Accumulate E[dx²]
	md_smul(d->dom->N, d->dom->dims, d->floating_dx, d->floating_dx, d->rho / (1. - d->rho));
	md_fmac(d->dom->N, d->dom->dims, d->floating_dx, dst, dst);
	md_smul(d->dom->N, d->dom->dims, d->floating_dx, d->floating_dx, (1. - d->rho));
}


static void adadelta_update_free(const operator_data_t* _data)
{
        const auto d = CAST_DOWN(adadelta_update_s, _data);
        iovec_free(d->dom);
	md_free(d->floating_g);
	md_free(d->floating_dx);
	xfree(d);
}

const struct operator_s* operator_adadelta_update_create(unsigned int N, const long dims[N], float lr, float rho, float epsilon)
{
	PTR_ALLOC(struct adadelta_update_s, data);
	SET_TYPEID(adadelta_update_s, data);

	long rdims[N + 1];
	rdims[0] = 2;
	md_copy_dims(N, rdims + 1, dims);

        data->dom = iovec_create(N + 1, rdims, FL_SIZE);
	data->floating_dx = NULL;
	data->floating_g = NULL;
	data->rho = rho;
	data->lr = lr;
	data->epsilon = epsilon;

        return operator_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), adadelta_update_apply, adadelta_update_free);
}


struct clip_s {

	INTERFACE(operator_data_t);

	const struct iovec_s* dom;

	float clipval;
	float clipnorm;
};

static DEF_TYPEID(clip_s);

static void clip_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	struct clip_s* d = CAST_DOWN(clip_s, _data);
	assert(2 == N);
	float* dst = (float*)args[0];
	float* src = (float*)args[1];

	if (d->clipnorm != 0.){

		float norm = md_norm(d->dom->N - 1, d->dom->dims + 1, src) / md_calc_size(d->dom->N, d->dom->dims);
		md_smul(d->dom->N, d->dom->dims, dst, src, (norm > d->clipnorm) ? (d->clipnorm / norm) : 1.);
	}

	if (d->clipval != 0.){

		md_smax(d->dom->N, d->dom->dims, dst, (d->clipnorm != 0.) ? dst : src, -d->clipval);
		md_smin(d->dom->N, d->dom->dims, dst, dst, d->clipval);
	}

}

static void clip_free(const operator_data_t* _data)
{
        const auto d = CAST_DOWN(clip_s, _data);
        iovec_free(d->dom);
	xfree(d);
}

const struct operator_s* operator_clip_create(unsigned int N, const long dims[N], float clipnorm, float clipval)
{
	PTR_ALLOC(struct clip_s, data);
	SET_TYPEID(clip_s, data);

	long rdims[N + 1];
	rdims[0] = 2;
	md_copy_dims(N, rdims + 1, dims);

        data->dom = iovec_create(N + 1, rdims, FL_SIZE);

	data->clipnorm = clipnorm;
	data->clipval = clipval;

        return operator_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), clip_apply, clip_free);
}