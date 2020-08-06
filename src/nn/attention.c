#include <complex.h>

#include "linops/someops.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/rand.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "nlops/nlop.h"
#include "nn/layers.h"

#include "attention.h"


struct self_attention_s {

	INTERFACE(nlop_data_t);

	unsigned long N;

	const long* dims; // [2, feature_size, query_size, P_x, P_y, P_z, A_x, A_y, A_z, Batch, Heads]

	const long* qstrs; //[x, 0, x, x, x, x, 0, 0, 0, x, x]
	const long* kstrs; //[x, 0, x, 0, 0, 0, x, x, x, x, x]
	const long* vstrs; //[x, x, 0, 0, 0, 0, x, x, x, x, x]
	const long* astrs; //[0, 0, 0, x, x, x, x, x, x, x, x]
	const long* rstrs; //[x, x, 0, x, x, x, 0, 0, 0, x, x]

	const long* qdims; //[x, 0, x, x, x, x, 0, 0, 0, x, x]
	const long* kdims; //[x, 0, x, 0, 0, 0, x, x, x, x, x]
	const long* vdims; //[x, x, 0, 0, 0, 0, x, x, x, x, x]
	const long* rdims; //[x, x, 0, x, x, x, 0, 0, 0, x, x]

	const long* kdims_red; //[x, 0, x, 0, 0, 0, 0, 0, 0, x, x]
	const long* vdims_red; //[x, x, 0, 0, 0, 0, 0, 0, 0, x, x]
	const long* adims_red; //[0, 0, 0, x, x, x, 0, 0, 0, x, x]
	const long* dims_red; //[x, x, x, x, x, x, 0, 0, 0, x, x]

	unsigned long red_flag;

	float*  negative_max_attention;
	float*  normalization;

	float* query;
	float* key;
	float* value;
	float* neg_result;
};

DEF_TYPEID(self_attention_s);

static void get_max_attention(struct self_attention_s* d, float* query, float* key)
{
	long N = d->N;

	if (NULL == d->negative_max_attention)
		d->negative_max_attention = md_alloc_sameplace(N, d->adims_red, FL_SIZE, query);

	float* tmp_att = md_alloc_sameplace(N, d->adims_red, FL_SIZE, query);

	bool first = true;
	long pos[N];
	memset(pos, 0, N * sizeof(long));
	do {
		md_clear(N, d->adims_red, tmp_att, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att, d->qstrs, query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, key));

		if (first) {

			first = false;
			md_copy(N, d->adims_red, d->negative_max_attention, tmp_att, FL_SIZE);
		} else {

			md_max(N, d->adims_red, d->negative_max_attention, d->negative_max_attention, tmp_att);
		}

	} while (md_next(N, d->dims, d->red_flag, pos));

	md_smul(N, d->adims_red, d->negative_max_attention, d->negative_max_attention, -1.);
	md_free(tmp_att);
}

static void compute_attention(struct self_attention_s* d, float* result, float* query, float* key, float* value)
{
	long N = d->N;

	if (NULL == d->negative_max_attention)
		get_max_attention(d, query, key);

	bool compute_normalization = (NULL == d->normalization);

	if (compute_normalization) {

		d->normalization = md_alloc_sameplace(N, d->adims_red, FL_SIZE, query);
		md_clear(N, d->adims_red, d->normalization, FL_SIZE);
	}

	float* tmp_att = md_alloc_sameplace(N, d->adims_red, FL_SIZE, query);

	md_clear(N, d->rdims, result, FL_SIZE);

	long pos[N];
	memset(pos, 0, N * sizeof(long));
	do {

		md_copy(N, d->adims_red, tmp_att, d->negative_max_attention, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att, d->qstrs, query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, key));
		md_exp(N, d->adims_red, tmp_att, tmp_att);

		if (compute_normalization)
			md_add(N, d->adims_red, d->normalization, d->normalization, tmp_att);

		md_fmac2(N, d->rdims, d->rstrs, result, d->astrs, tmp_att, d->vstrs, &MD_ACCESS(N, d->vstrs, pos, value));

	} while (md_next(N, d->dims, d->red_flag, pos));

	if (compute_normalization) {

		float* tmp = md_alloc_sameplace(N, d->adims_red, FL_SIZE, query);
		float one = 1.;
		md_fill(N, d->adims_red, tmp, &one, FL_SIZE);
		md_div(N, d->adims_red, d->normalization, tmp, d->normalization);
		md_free(tmp);
	}


	md_mul2(N, d->rdims, d->rstrs, result, d->rstrs, result, d->astrs, d->normalization);
	md_free(tmp_att);
}

static void compute_attention_adj(struct self_attention_s* d, float* result, float* query, float* key, float* value)
{
	long N = d->N;

	float* tmp_result = md_alloc_sameplace(N, d->rdims, FL_SIZE, query);
	md_mul2(N, d->rdims, d->rstrs, tmp_result, d->rstrs, result, d->astrs, d->normalization);

	float* tmp_att = md_alloc_sameplace(N, d->adims_red, FL_SIZE, query);

	md_clear(N, d->vdims, value, FL_SIZE);

	long pos[N];
	memset(pos, 0, N * sizeof(long));
	do {

		md_copy(N, d->adims_red, tmp_att, d->negative_max_attention, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att, d->qstrs, query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, key));
		md_exp(N, d->adims_red, tmp_att, tmp_att);

		md_fmac2(N, d->rdims, d->vstrs, &MD_ACCESS(N, d->vstrs, pos, value), d->rstrs, tmp_result, d->astrs, tmp_att);

	} while (md_next(N, d->dims, d->red_flag, pos));

	md_free(tmp_att);
}

static void self_attention_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	START_TIMER;
	auto d= CAST_DOWN(self_attention_s, _data);

	assert(4 == N);
	float* result = (float*)args[0];
	float* query = (float*)args[1];
	float* key = (float*)args[2];
	float* value = (float*)args[3];

	N = d->N;

#ifdef USE_CUDA
	for (int i = 1; i < 4; i++)
		assert((cuda_ondevice(args[0]) == cuda_ondevice(args[i])));
#endif

	if (NULL != d->negative_max_attention)
		md_free(d->negative_max_attention);
	if (NULL != d->normalization)
		md_free(d->normalization);

	d->negative_max_attention = NULL;
	d->normalization = NULL;

	if (NULL == d->neg_result)
		d->neg_result = md_alloc_sameplace(N, d->rdims, FL_SIZE, result);
	if (NULL == d->query)
		d->query = md_alloc_sameplace(N, d->qdims, FL_SIZE, result);
	if (NULL == d->key)
		d->key = md_alloc_sameplace(N, d->kdims, FL_SIZE, result);
	if (NULL == d->value)
		d->value = md_alloc_sameplace(N, d->vdims, FL_SIZE, result);

	md_copy(N, d->qdims, d->query, query, FL_SIZE);
	md_copy(N, d->kdims, d->key, key, FL_SIZE);
	md_copy(N, d->vdims, d->value, value, FL_SIZE);

	compute_attention(d, result, query, key, value);

	md_smul(N, d->rdims, d->neg_result, result, -1.);

	PRINT_TIMER("frw self attention");
}

static void self_attention_der_val(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	auto d = CAST_DOWN(self_attention_s, _data);

	compute_attention(d, (float*)dst, d->query, d->key, (float*)src);

	PRINT_TIMER("der val self attention");
}

static void self_attention_adj_val(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	auto d = CAST_DOWN(self_attention_s, _data);

	compute_attention_adj(d, (float*)src, d->query, d->key, (float*)dst);

	PRINT_TIMER("adj val self attention");
}

static void self_attention_der_key(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	auto d = CAST_DOWN(self_attention_s, _data);

	int N = d->N;

	float* tmp_att = md_alloc_sameplace(N, d->adims_red, FL_SIZE, d->query);
	float* tmp_att2 = md_alloc_sameplace(N, d->adims_red, FL_SIZE, d->query);

	md_clear(N, d->rdims, dst, FL_SIZE);

	long pos[N];
	memset(pos, 0, N * sizeof(long));
	do {

		md_copy(N, d->adims_red, tmp_att, d->negative_max_attention, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att, d->qstrs, d->query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, d->key));
		md_exp(N, d->adims_red, tmp_att, tmp_att);

		md_clear(N, d->adims_red, tmp_att2, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att2, d->qstrs, d->query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, (float*)src));

		md_mul(N, d->adims_red, tmp_att, tmp_att, tmp_att2);

		md_fmac2(N, d->rdims, d->rstrs, (float*)dst, d->astrs, tmp_att, d->vstrs, &MD_ACCESS(N, d->vstrs, pos, d->value));
		md_fmac2(N, d->rdims, d->rstrs, (float*)dst, d->astrs, tmp_att, d->rstrs, d->neg_result);

	} while (md_next(N, d->dims, d->red_flag, pos));

	md_mul2(N, d->rdims, d->rstrs, (float*)dst, d->rstrs, (float*)dst, d->astrs, d->normalization);
	md_free(tmp_att);
	md_free(tmp_att2);

	PRINT_TIMER("der key self attention");
}

static void self_attention_adj_key(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	auto d = CAST_DOWN(self_attention_s, _data);

	int N = d->N;

	float* tmp_z = md_alloc_sameplace(N, d->rdims, FL_SIZE, d->query);
	md_mul2(N, d->rdims, d->rstrs, tmp_z, d->rstrs, (float*)src, d->astrs, d->normalization);

	float* tmp_z_sa = md_alloc_sameplace(N, d->rdims, FL_SIZE, d->query);
	md_mul(N, d->rdims, tmp_z_sa, tmp_z, d->neg_result);

	float* tmp_att = md_alloc_sameplace(N, d->adims_red, FL_SIZE, d->query);
	float* tmp_z_sa_v = md_alloc_sameplace(N, d->rdims, FL_SIZE, d->query);

	md_clear(N, d->kdims, dst, FL_SIZE);
	debug_print_dims(DP_INFO, N, d->kdims);

	long pos[N];
	memset(pos, 0, N * sizeof(long));

	do {
		md_copy(N, d->adims_red, tmp_att, d->negative_max_attention, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att, d->qstrs, d->query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, d->key));
		md_exp(N, d->adims_red, tmp_att, tmp_att);

		md_copy(N, d->rdims, tmp_z_sa_v, tmp_z_sa, FL_SIZE);
		md_fmac2(N, d->rdims, d->rstrs, tmp_z_sa_v, d->rstrs, tmp_z, d->vstrs, &MD_ACCESS(N, d->vstrs, pos, d->value));

		md_mul2(N, d->rdims, d->rstrs, tmp_z_sa_v, d->rstrs, tmp_z_sa_v, d->astrs, tmp_att);
		md_fmac2(N, d->dims_red, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, (float*)dst), d->qstrs, d->query, d->rstrs, tmp_z_sa_v);

	} while (md_next(N, d->dims, d->red_flag, pos));


	md_free(tmp_att);
	md_free(tmp_z);
	md_free(tmp_z_sa);
	md_free(tmp_z_sa_v);

	PRINT_TIMER("adj key self attention");
}



static void self_attention_der_query(const nlop_data_t* _data, complex float* dst, const complex float* src)
{

	START_TIMER;
	auto d = CAST_DOWN(self_attention_s, _data);

	int N = d->N;

	float* tmp_att2 = md_alloc_sameplace(N, d->adims_red, FL_SIZE, d->query);
	float* tmp_att = md_alloc_sameplace(N, d->adims_red, FL_SIZE, d->query);

	md_clear(N, d->rdims, dst, FL_SIZE);

	long pos[N];
	memset(pos, 0, N * sizeof(long));
	do {

		md_copy(N, d->adims_red, tmp_att, d->negative_max_attention, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att, d->qstrs, d->query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, d->key));
		md_exp(N, d->adims_red, tmp_att, tmp_att);

		md_clear(N, d->adims_red, tmp_att2, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att2, d->qstrs, (float*)src, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, d->key)); //Q*K

		md_mul(N, d->adims_red, tmp_att, tmp_att, tmp_att2); //smax*Q*K

		md_fmac2(N, d->rdims, d->rstrs, (float*)dst, d->astrs, tmp_att, d->vstrs, &MD_ACCESS(N, d->vstrs, pos, d->value));
		md_fmac2(N, d->rdims, d->rstrs, (float*)dst, d->astrs, tmp_att, d->rstrs, d->neg_result);

	} while (md_next(N, d->dims, d->red_flag, pos));

	md_mul2(N, d->rdims, d->rstrs, (float*)dst, d->rstrs, (float*)dst, d->astrs, d->normalization);

	md_free(tmp_att);
	md_free(tmp_att2);

	PRINT_TIMER("der query self attention");
}

static void self_attention_adj_query(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	START_TIMER;
	auto d = CAST_DOWN(self_attention_s, _data);

	int N = d->N;

	float* tmp_z = md_alloc_sameplace(N, d->rdims, FL_SIZE, d->query);
	md_mul2(N, d->rdims, d->rstrs, tmp_z, d->rstrs, (float*)src, d->astrs, d->normalization);

	float* tmp_z2 = md_alloc_sameplace(N, d->rdims, FL_SIZE, d->query);
	float* tmp_z3 = md_alloc_sameplace(N, d->rdims, FL_SIZE, d->query);

	float* tmp_att = md_alloc_sameplace(N, d->adims_red, FL_SIZE, d->query);


	md_clear(N, d->qdims, dst, FL_SIZE);

	long pos[N];
	memset(pos, 0, N * sizeof(long));
	do {

		md_copy(N, d->adims_red, tmp_att, d->negative_max_attention, FL_SIZE);
		md_fmac2(N, d->qdims, d->astrs, tmp_att, d->qstrs, d->query, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, d->key));
		md_exp(N, d->adims_red, tmp_att, tmp_att);

		md_mul2(N, d->rdims, d->rstrs, tmp_z2, d->rstrs, tmp_z, d->astrs, tmp_att);
		md_mul(N, d->rdims, tmp_z3, tmp_z2, d->neg_result);
		md_fmac2(N, d->rdims, d->rstrs, tmp_z3, d->rstrs, tmp_z2, d->vstrs, &MD_ACCESS(N, d->vstrs, pos, d->value));

		md_fmac2(N, d->dims_red, d->qstrs, (float*)dst, d->kstrs, &MD_ACCESS(N, d->kstrs, pos, d->key), d->rstrs, tmp_z3);

	} while (md_next(N, d->dims, d->red_flag, pos));

	md_free(tmp_z);
	md_free(tmp_z2);
	md_free(tmp_att);
	md_free(tmp_z3);

	PRINT_TIMER("adj query self attention");
}


static void self_attention_del(const struct nlop_data_s* _data)
{
	const auto d = CAST_DOWN(self_attention_s, _data);

	xfree(d->dims);

	xfree(d->qstrs);
	xfree(d->kstrs);
	xfree(d->vstrs);
	xfree(d->astrs);
	xfree(d->rstrs);

	xfree(d->qdims);
	xfree(d->kdims);
	xfree(d->vdims);
	xfree(d->rdims);

	xfree(d->kdims_red);
	xfree(d->vdims_red);
	xfree(d->adims_red);
	xfree(d->dims_red);

	md_free(d->negative_max_attention);
	md_free(d->normalization);

	md_free(d->query);
	md_free(d->key);
	md_free(d->value);
	md_free(d->neg_result);

	xfree(d);
}


/***
 * This creates a self-attention module
 *
 *Inputs:
 *	Query: dims = [k, px, py, pz, batch]
 *	Key: dims = [k, ax, ay, az, batch]
 *	Value: dims = [f, az, ay, az, batch]
 * Outputs:
 *	dims = [f, px, py, pz, batch]
 *	sum_a softmax_a(sum_k Query * Key) * value
 *
 * if heads != 1, k and f are divided by head and operator is also batched over the heads dim
 */

const struct nlop_s* nlop_self_attention_create(int N, const long qdims[N], const long kdims[N], const long vdims[N], long heads)
{
	error("The self-attention operator does not work yet, the adjoint test is not passed!\n");
	//dims [4, 128, 128, 1, 10] take about 5sec on GeForce GTX Titan (Radon 2)

#if 0
static bool test_nlop_attention(void)
{
 	enum { N = 5 };
 	long dims[N] = { 4, 3, 3, 1, 1};

	auto op = nlop_self_attention_create(N, dims, dims, dims, 1);

	float err_adj = nlop_test_adj_derivatives(op, true);
	float err_der = nlop_test_derivatives(op);

	debug_printf(DP_INFO, "self-attention errors der, adj: %.8f, %.8f\n", err_der, err_adj);
	UT_ASSERT((err_der < 1.E-2) && (err_adj < 1.E-6));
}
#endif

	PTR_ALLOC(struct self_attention_s, d);
	SET_TYPEID(self_attention_s, d);

	assert(5 == N);
	assert(md_check_equal_dims(N, kdims, qdims, ~0ul));
	assert(md_check_equal_dims(N, kdims, vdims, ~1ul));

	assert(0 == kdims[0] % heads);
	assert(0 == vdims[0] % heads);

	N = 11;
	d->N = N;
	long dims[11] = {2, vdims[0] / heads, qdims[0] / heads, vdims[1], vdims[2], vdims[3], vdims[1], vdims[2], vdims[3], vdims[4], heads};

	unsigned long qflags = MD_BIT(0) | MD_BIT(2) | MD_BIT(3) | MD_BIT(4) | MD_BIT(5) | MD_BIT(9) | MD_BIT(10);
	unsigned long kflags = MD_BIT(0) | MD_BIT(2) | MD_BIT(6) | MD_BIT(7) | MD_BIT(8) | MD_BIT(9) | MD_BIT(10);
	unsigned long vflags = MD_BIT(0) | MD_BIT(1) | MD_BIT(6) | MD_BIT(7) | MD_BIT(8) | MD_BIT(9) | MD_BIT(10);
	unsigned long aflags = MD_BIT(3) | MD_BIT(4) | MD_BIT(5) | MD_BIT(6) | MD_BIT(7) | MD_BIT(8) | MD_BIT(9) | MD_BIT(10);
	unsigned long rflags = MD_BIT(0) | MD_BIT(1) | MD_BIT(3) | MD_BIT(4) | MD_BIT(5) | MD_BIT(9) | MD_BIT(10);

	d->red_flag = MD_BIT(6) | MD_BIT(7) | MD_BIT(8);

	PTR_ALLOC(long[N], ndims);
	PTR_ALLOC(long[N], nqdims);
	PTR_ALLOC(long[N], nkdims);
	PTR_ALLOC(long[N], nvdims);
	PTR_ALLOC(long[N], nadims);
	PTR_ALLOC(long[N], nrdims);

	PTR_ALLOC(long[N], nkdims_red);
	PTR_ALLOC(long[N], nvdims_red);
	PTR_ALLOC(long[N], nadims_red);
	PTR_ALLOC(long[N], ndims_red);

	md_select_dims(N, ~0ul, *ndims, dims);
	md_select_dims(N, qflags, *nqdims, dims);
	md_select_dims(N, kflags, *nkdims, dims);
	md_select_dims(N, vflags, *nvdims, dims);
	md_select_dims(N, aflags, *nadims, dims);
	md_select_dims(N, rflags, *nrdims, dims);

	md_select_dims(N, kflags & (~d->red_flag), *nkdims_red, dims);
	md_select_dims(N, vflags & (~d->red_flag), *nvdims_red, dims);
	md_select_dims(N, aflags & (~d->red_flag), *nadims_red, dims);
	md_select_dims(N, (~d->red_flag), *ndims_red, dims);

	d->dims = *PTR_PASS(ndims);
	d->qdims = *PTR_PASS(nqdims);
	d->kdims = *PTR_PASS(nkdims);
	d->vdims = *PTR_PASS(nvdims);
	d->rdims = *PTR_PASS(nrdims);

	d->kdims_red = *PTR_PASS(nkdims_red);
	d->vdims_red = *PTR_PASS(nvdims_red);
	d->adims_red = *PTR_PASS(nadims_red);
	d->dims_red = *PTR_PASS(ndims_red);

	PTR_ALLOC(long[N], nqstrs);
	PTR_ALLOC(long[N], nkstrs);
	PTR_ALLOC(long[N], nvstrs);
	PTR_ALLOC(long[N], nastrs);
	PTR_ALLOC(long[N], nrstrs);

	md_calc_strides(N, *nqstrs, d->qdims, FL_SIZE);
	md_calc_strides(N, *nkstrs, d->kdims, FL_SIZE);
	md_calc_strides(N, *nvstrs, d->vdims, FL_SIZE);
	md_calc_strides(N, *nastrs, d->adims_red, FL_SIZE);
	md_calc_strides(N, *nrstrs, d->rdims, FL_SIZE);

	d->qstrs = *PTR_PASS(nqstrs);
	d->kstrs = *PTR_PASS(nkstrs);
	d->vstrs = *PTR_PASS(nvstrs);
	d->astrs = *PTR_PASS(nastrs);
	d->rstrs = *PTR_PASS(nrstrs);

	d->negative_max_attention = NULL;
	d->normalization = NULL;

	d->query = NULL;
	d->key = NULL;
	d->value = NULL;
	d->neg_result = NULL;

	long nl_idims[3][(1 == heads) ? 5 : 6];
	md_copy_dims(5, nl_idims[0], qdims);
	md_copy_dims(5, nl_idims[1], kdims);
	md_copy_dims(5, nl_idims[2], vdims);

	long nl_odims[3][(1 == heads) ? 5 : 6];
	md_copy_dims(5, nl_odims[0], vdims);

	if (1 != heads) {

		nl_idims[0][0] /= heads;
		nl_idims[1][0] /= heads;
		nl_idims[2][0] /= heads;
		nl_odims[0][0] /= heads;

		nl_idims[0][5] = heads;
		nl_idims[0][5] = heads;
		nl_idims[0][5] = heads;
		nl_odims[0][5] = heads;
	}

	const struct nlop_s* result = nlop_generic_create(	1, (1 == heads) ? 5 : 6, nl_odims,
								3, (1 == heads) ? 5 : 6, nl_idims,
								CAST_UP(PTR_PASS(d)),
								self_attention_fun,
								(nlop_fun_t[3][1]){ { self_attention_der_query }, { self_attention_der_key }, { self_attention_der_val } },
								(nlop_fun_t[3][1]){ { self_attention_adj_query }, { self_attention_adj_key }, { self_attention_adj_val } },
								NULL, NULL, self_attention_del);

	if (1 != heads) {

		//FIXME: implement linop_permute, splt features in features and heads. permute heads to last dimension and chain into attention block
		error("Multiple attention heads not implemented");
	}

	return result;
}
