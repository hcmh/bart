
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"
#include "num/ops_p.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/mri.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/stack.h"
#include "nlops/const.h"

#include "moba/scale.h"


#include "utest.h"



// Create a simplisti coperator in the nlop interface, which is required for the scaling function
// (!) The implemented operator here is linear, but tests if the scaling function works accurately.
// f(x,y) = x + 2 y
struct testFun_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* map_dims;
	const long* in_dims;
	const long* out_dims;

	const long* strs;
	const long* map_strs;
	const long* in_strs;
	const long* out_strs;

	complex float* x;
	complex float* y;
	complex float* tmp_map;
	complex float* derivatives;
};

DEF_TYPEID(testFun_s);


static void test_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct testFun_s* data = CAST_DOWN(testFun_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// Clean up
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);

	// Signal

	md_copy_block(data->N, pos, data->map_dims, data->x, data->in_dims, src, CFL_SIZE);
	
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->y, data->in_dims, src, CFL_SIZE);


	md_zaxpy(data->N, data->map_dims, data->tmp_map, 2., data->y);//2.

	md_zadd(data->N, data->map_dims, dst, data->x, data->tmp_map);

	// Clean up
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);

	// Derivatives

	md_clear(data->N, data->in_dims, data->derivatives, CFL_SIZE);

	md_set_dims(data->N, pos, 0);

	md_zfill(data->N, data->map_dims, data->tmp_map, 1.);
	pos[COEFF_DIM] = 0; // x
	md_copy_block(data->N, pos, data->in_dims, data->derivatives, data->map_dims, data->tmp_map, CFL_SIZE);

	md_zfill(data->N, data->map_dims, data->tmp_map, 2.);//2.
	pos[COEFF_DIM] = 1; // y
	md_copy_block(data->N, pos, data->in_dims, data->derivatives, data->map_dims, data->tmp_map, CFL_SIZE);

	// char name[255] = {'\0'};
	// sprintf(name, "_map2");
	// dump_cfl(name, data->N, data->in_dims, data->derivatives);
}


static void test_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	// debug_printf(DP_INFO, "Start Derivative\n");

	struct testFun_s* data = CAST_DOWN(testFun_s, _data);

	md_clear(data->N, data->out_dims, dst, CFL_SIZE);

	md_ztenmul(data->N, data->map_dims, dst, data->in_dims, data->derivatives, data->in_dims, src);

}



static void test_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	// debug_printf(DP_INFO, "Start Adjoint\n");

	struct testFun_s* data = CAST_DOWN(testFun_s, _data);

	md_clear(data->N, data->in_dims, dst, CFL_SIZE);

	md_zfmacc2(data->N, data->in_dims, data->in_strs, dst, data->map_strs, src, data->in_strs, data->derivatives);
}


static void test_del(const nlop_data_t* _data)
{
	struct testFun_s* data = CAST_DOWN(testFun_s, _data);

	md_free(data->x);
	md_free(data->y);
	md_free(data->tmp_map);
	md_free(data->derivatives);

	xfree(data->map_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->strs);
	xfree(data->map_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);
}


static struct nlop_s* nlop_test_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N])
{
	PTR_ALLOC(struct testFun_s, data);
	SET_TYPEID(testFun_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(long[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nmstr);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr);

	PTR_ALLOC(long[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
	data->in_strs = *PTR_PASS(nistr);

	data->N = N;

	data->x = md_alloc(N, map_dims, CFL_SIZE);
	data->y = md_alloc(N, map_dims, CFL_SIZE);
	data->tmp_map = md_alloc(N, map_dims, CFL_SIZE);
	data->derivatives = md_alloc(N, in_dims, CFL_SIZE);

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), test_fun, test_der, test_adj, NULL, NULL, test_del);
}




static bool test_nlop_testfun(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* dst = md_alloc(N, out_dims, CFL_SIZE);
	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	md_zfill(N, in_dims, src, 1.0);

	struct nlop_s* test = nlop_test_create(N, map_dims, out_dims, in_dims);

	nlop_apply(test, N, out_dims, dst, N, in_dims, src);

	float err = linop_test_adjoint(nlop_get_derivative(test, 0, 0));

	nlop_free(test);
	md_free(src);
	md_free(dst);

	UT_ASSERT(err < 1.E-3);
}
UT_REGISTER_TEST(test_nlop_testfun);



/* Test operator covers function f(x,y) = x + 2* y
Normal operator:
	A=DF^H(x)DF(x) = {{1,2},{2,4}}
Projection:
	P1={{1,0},{0,0}},	P1={{0,0},{0,1}}

Max. Eigenvalue(P1 A) -> 1
Max. Eigenvalue(P2 A) -> 4
*/

static bool test_nlop_op_ev(void)
{
	enum { N = 16 };
	long map_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long out_dims[N] = { 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	long in_dims[N] = { 16, 16, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	complex float* src = md_alloc(N, in_dims, CFL_SIZE);

	md_zfill(N, in_dims, src, 1.0);

	complex float* ev = md_alloc(1, MD_DIMS(in_dims[COEFF_DIM]), CFL_SIZE);

	// f(x,y) = x + 2 y
	struct nlop_s* test = nlop_test_create(N, map_dims, out_dims, in_dims);

	nlop_get_partial_ev(test, in_dims, ev, src);

	UT_ASSERT(crealf(ev[0]) == 1.);

	UT_ASSERT(crealf(ev[1]) == 4.);

	nlop_free(test);

	md_free(src);
	md_free(ev);

}
UT_REGISTER_TEST(test_nlop_op_ev);
