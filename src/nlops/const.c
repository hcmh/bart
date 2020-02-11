#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "const.h"
#include "misc/debug.h"


struct const_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	complex float* xn;
};

DEF_TYPEID(const_s);

static void const_fun(const nlop_data_t* _data, int N, complex float** dst)
{
	UNUSED(N);
	const auto data = CAST_DOWN(const_s, _data);

	md_copy(data->N, data->dims, dst[0], data->xn, CFL_SIZE);
}

static void const_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(const_s, _data);

	md_free(data->xn);
	xfree(data->dims);
	xfree(data);
}

/**
 * Create operator with constant output (zero inputs, one output)
 * Strides are only applied on the input
 * @param N #dimensions
 * @param dims dimensions
 * @param strs in-strides
 * @param in reference to constant input array
 */
struct nlop_s* nlop_const_create2(int N, const long dims[N], const long strs[N], const complex float* in)
{
	PTR_ALLOC(struct const_s, data);
	SET_TYPEID(const_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->xn = md_alloc(N, dims, CFL_SIZE);

	md_copy2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), data->xn, strs, in, CFL_SIZE);

	long ostrs[N];
	md_calc_strides(N, ostrs, dims, CFL_SIZE);

	long tdims[1][N];
	md_copy_dims(N, tdims[0], dims);

	return nlop_generic_create(1, N, tdims, 0, 0, NULL, CAST_UP(PTR_PASS(data)), const_fun, NULL, NULL, NULL,NULL, const_del);
}

/**
 * Create operator with constant output (zero inputs, one output)
 * @param N #dimensions
 * @param dims dimensions
 * @param in reference to constant input array
 */
struct nlop_s* nlop_const_create(int N, const long dims[N], const complex float* in)
{
	return nlop_const_create2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), in);
}


/**
 * Chain operator with a constant operator
 * @param a operator whose input should be set constant 
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param strs strides of input array
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const2(const struct nlop_s* a, int i, int N, const long dims[N], const long strs[N], const complex float* in)
{
	int ai = nlop_get_nr_in_args(a);
	assert(i < ai);

	struct nlop_s* nlop_const = nlop_const_create2(N, dims, strs, in);
	struct nlop_s* result = nlop_chain2(nlop_const, 0,  a,  i);
	nlop_free(nlop_const);

	return result;
}

/**
 * Chain operator with a constant operator
 * @param a operator whose input should be set constant 
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const(const struct nlop_s* a, int i, int N, const long dims[N], const complex float* in)
{
	return nlop_set_input_const2(a, i, N, dims, MD_STRIDES(N, dims, CFL_SIZE), in);
}

/**
 * Chain operator with a constant operator and free the input operator
 * @param a operator whose input should be set constant 
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param strs strides of input array
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const_F2(const struct nlop_s* a, int i, int N, const long dims[N], const long strs[N], const complex float* in)
{
	struct nlop_s* result = nlop_set_input_const2(a, i, N, dims, strs, in);
	nlop_free(a);
	return result;
}

/**
 * Chain operator with a constant operator and free the input operator
 * @param a operator whose input should be set constant 
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const_F(const struct nlop_s* a, int i, int N, const long dims[N], const complex float* in)
{
    struct nlop_s* result = nlop_set_input_const(a, i, N, dims, in);
    nlop_free(a);
	return result;
}

struct del_out_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(del_out_s);

static void del_out_fun(const nlop_data_t* _data, int N, complex float** in)
{
	UNUSED(N);
	UNUSED(_data);
	UNUSED(in);
}

static void del_out_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(del_out_s, _data);
	xfree(data);
}

/**
 * Create operator with one input and zero outputs
 * @param N #dimensions
 * @param dims dimensions
 */
struct nlop_s* nlop_del_out_create(int N, const long dims[N])
{
	PTR_ALLOC(struct del_out_s, data);
	SET_TYPEID(del_out_s, data);

	long tdims[1][N];
	md_copy_dims(N, tdims[0], dims);

	return nlop_generic_create(0, 0, NULL, 1, N, tdims, CAST_UP(PTR_PASS(data)), del_out_fun, NULL, NULL, NULL,NULL, del_out_del);
}

/**
 * Returns a new operator without the output o
 * @param a operator
 * @param o index of output to be deleted
 */
struct nlop_s* nlop_del_out(const struct nlop_s* a, int o)
{
	int ao = nlop_get_nr_out_args(a);
	assert(ao > o);

	const struct iovec_s* codomain = nlop_generic_codomain(a, o);
    
	struct nlop_s* nlop_del_out_op = nlop_del_out_create(codomain->N, codomain->dims);
	struct nlop_s* result = nlop_chain2(a, o,  nlop_del_out_op,  0);
	nlop_free(nlop_del_out_op);

	return result;
}
