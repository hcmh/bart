#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/shrdptr.h"
#include "misc/types.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "initializer.h"

typedef void (*initializer_del)(const struct initializer_s* conf);

typedef struct initializer_s{

	TYPEID* TYPEID;

	initializer_f fun;
	initializer_del del;
	struct shared_obj_s sptr;

} init_t;

static void init_del(const struct shared_obj_s* sptr)
{
	const struct initializer_s* x = CONTAINER_OF(sptr, const struct initializer_s, sptr);

	if (NULL != x->del)
		x->del(x);
	xfree(x);
}

void initializer_free(const struct initializer_s* x)
{
	if (NULL == x)
		return;

	shared_obj_destroy(&x->sptr);
}

const struct initializer_s* initializer_clone(const struct initializer_s* x)
{
	if (NULL != x)
		shared_obj_ref(&x->sptr);

	return x;
}

void initializer_apply(const struct initializer_s* x, long N, const long dims[N], complex float* weights)
{
	x->fun(x, N, dims, weights);
}


unsigned long in_flag_conv(bool c1)
{
	unsigned long in_flags = c1 ? MD_BIT(1) : MD_BIT(3);

	//filters, channel, kx, ky, kz    or x, y, z channel, filters
	for (int i = 0; i < 3; i++)
		in_flags |= MD_BIT(i + (c1 ? 0 : 2));
	return in_flags;
}

unsigned long out_flag_conv(bool c1)
{
	unsigned long out_flags = c1 ? MD_BIT(0) : MD_BIT(4);

	//filters, channel, kx, ky, kz    or x, y, z channel, filters
	for (int i = 0; i < 3; i++)
		out_flags |= MD_BIT(i + (c1 ? 0 : 2));
	return out_flags;
}

struct initializer_const_s {

	INTERFACE(init_t);
	complex float val;
};

static DEF_TYPEID(initializer_const_s);

static void init_const_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_const_s, conf_);
	md_zfill(N, dims, weights, conf->val);
}

const struct initializer_s* init_const_create(_Complex float val)
{
	PTR_ALLOC(struct initializer_const_s, data);
	SET_TYPEID(initializer_const_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_const_fun;

	data->val = val;

	return CAST_UP(PTR_PASS(data));
}

// Returns real/complex uniform/normal distribution with mean 0 and variance 1
static void get_base_dist(unsigned int N, const long dims[N], complex float* dst, bool uniform, bool real)
{
	if (uniform) {

		md_uniform_rand(N, dims, dst);
		md_zsadd(N, dims, dst, dst, (complex float)(-0.5));

		if (!real) {

			complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
			md_uniform_rand(N, dims, tmp);
			md_zsadd(N, dims, tmp, tmp, (complex float)(-0.5));
			md_zaxpy(N, dims, dst, 1.I, tmp);
			md_free(tmp);
		}

		md_zsmul(N, dims, dst, dst, real ? sqrt(12) : sqrt(6));
	} else {

		md_gaussian_rand(N, dims, dst);
		if (real)
			md_zreal(N, dims, dst, dst);
		else
			md_zsmul(N, dims, dst, dst, 1. / sqrt(2.));
	}
}

/*
Xavier Glorot, Yoshua Bengio ; Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, JMLR Workshop and Conference Proceedings 9:249-256, 2010.
Glorot, X. & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, in PMLR 9:249-256
*/
static float get_scaling_xavier(unsigned int N, const long dims[N], unsigned long in_flags, unsigned long out_flags)
{
	long tdims[N];
	md_select_dims(N, in_flags, tdims, dims);
	long inputs = md_calc_size(N, tdims);
	md_select_dims(N, out_flags, tdims, dims);
	long outputs = md_calc_size(N, tdims);

	return (float)sqrt(2. / (double)(inputs + outputs));
}

/*
He, K.; Zhang, X.; Ren, S. & Sun, J.
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
2015
*/
static float get_scaling_kaiming(unsigned int N, const long dims[N], unsigned long in_flags, float leaky_val)
{
	long tdims[N];
	md_select_dims(N, in_flags, tdims, dims);
	long inputs = md_calc_size(N, tdims);

	return (float)sqrt(2. / (double)(inputs) / (1. + leaky_val * leaky_val));
}

struct initializer_xavier_kaiming_s {

	INTERFACE(init_t);

	bool uniform;
	bool real;

	unsigned long in_flags;
	unsigned long out_flags;

	float leaky_val;
};

static DEF_TYPEID(initializer_xavier_kaiming_s);

static void init_xavier_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_xavier_kaiming_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, get_scaling_xavier(N, dims, conf->in_flags, conf->out_flags));
}

static void init_kaiming_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_xavier_kaiming_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, get_scaling_kaiming(N, dims, conf->in_flags, conf->leaky_val));
}


const struct initializer_s* init_xavier_create(unsigned long in_flags, unsigned long out_flags, bool real, bool uniform)
{
	PTR_ALLOC(struct initializer_xavier_kaiming_s, data);
	SET_TYPEID(initializer_xavier_kaiming_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_xavier_fun;

	data->in_flags = in_flags;
	data->out_flags = out_flags;
	data->uniform = uniform;
	data->real = real;
	data->leaky_val = 1;

	return CAST_UP(PTR_PASS(data));
}

const struct initializer_s* init_kaiming_create(unsigned long in_flags, bool real, bool uniform, float leaky_val)
{
	PTR_ALLOC(struct initializer_xavier_kaiming_s, data);
	SET_TYPEID(initializer_xavier_kaiming_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_kaiming_fun;

	data->in_flags = in_flags;
	data->out_flags = 0;
	data->uniform = uniform;
	data->real = real;
	data->leaky_val = leaky_val;

	return CAST_UP(PTR_PASS(data));
}

