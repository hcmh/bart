#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "num/ops.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "checkpointing.h"


struct checkpoint_s {

	INTERFACE(nlop_data_t);

	const struct nlop_s* nlop;

	const struct op_options_s* opts_no_der;

	bool der_once;
	
	unsigned int II;
	unsigned int OO;

	unsigned int* DI;
	const long** idims;
	complex float** inputs;

	complex float** adj_in;
	complex float** der_in;

	complex float** adj_out;
	complex float** der_out;
	
	unsigned int* DO;
	const long** odims;
};

DEF_TYPEID(checkpoint_s);

static void checkpoint_free_der(struct checkpoint_s* d)
{
	for (uint i = 0; i < d->II * d->OO; i++) {

		md_free(d->adj_out[i]);
		md_free(d->der_out[i]);

		d->adj_out[i] = NULL;
		d->der_out[i] = NULL;
	}

	for (uint i = 0; i < d->OO; i++) {

		md_free(d->adj_in[i]);
		d->adj_in[i] = NULL;
	}

	for (uint i = 0; i < d->II; i++) {

		md_free(d->der_in[i]);
		d->der_in[i] = NULL;
	}
}


static void checkpoint_save_inputs(struct checkpoint_s* data, unsigned int II, const complex float* inputs[II], bool save_inputs)
{
	assert(II == data->II);
	assert(0 < II);
	for (uint i = 0; i < data->II; i++) {

		if (save_inputs && (NULL == data->inputs[i]))
			data->inputs[i] = md_alloc_sameplace(data->DI[i], data->idims[i], CFL_SIZE, inputs[0]);

		if (!save_inputs){

			md_free(data->inputs[i]);
			data->inputs[i] = NULL;
		} else {

			md_copy(data->DI[i], data->idims[i], data->inputs[i], inputs[i], CFL_SIZE);
		}
	}
}

static void checkpoint_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(checkpoint_s, _data);
	assert(data->II + data->OO == (uint)N);

	bool der = false;
	for (uint i = 0; i < data->II; i++)
		for (uint o = 0; o < data->OO; o++)
			der = der || !op_options_is_set_io(_data->options, o, i, OP_APP_NO_DER);

	checkpoint_save_inputs(data, data->II, (const complex float**)(args + data->OO), der);

	operator_option_flags_t opts_flags[data->OO][data->II];
	for (uint i = 0; i < data->II; i++)
		for (uint o = 0; o < data->OO; o++)
			opts_flags[o][i] = MD_BIT(OP_APP_NO_DER);
	
	const struct op_options_s* opts = op_options_io_create(data->OO, data->II, operator_get_io_flags(data->nlop->op), opts_flags);
	nlop_generic_apply_with_opts_unchecked(data->nlop, N, (void**)args, opts);
	op_options_free(opts);

	for (uint i = 0; i < data->II; i++)
		for (uint o = 0; o < data->OO; o++)
			opts_flags[o][i] = op_options_is_set_io(_data->options, o, i, OP_APP_NO_DER) ? MD_BIT(OP_APP_NO_DER) : 0;

	op_options_free(data->opts_no_der);
	data->opts_no_der = op_options_io_create(data->OO, data->II, operator_get_io_flags(data->nlop->op), opts_flags);

	checkpoint_free_der(data);
}

static void checkpoint_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);
	
	if ((NULL != d->der_in[i]) && (1.e-8 > md_zrmse(d->DI[i], d->idims[i], d->der_in[i], src))) {

		assert(NULL != d->der_out[i + d->II * o]);
		md_copy(d->DO[o], d->odims[o], dst, d->der_out[i + d->II * o], CFL_SIZE);
		if (d->der_once) {

			md_free(d->der_out[i + d->II * o]);
			d->der_out[i + d->II * o] = NULL;
		}
		return;
	}

	if (NULL == d->der_in[i])
		d->der_in[i] = md_alloc_sameplace(d->DI[i], d->idims[i], CFL_SIZE, src);
	md_copy(d->DI[i], d->idims[i], d->der_in[i], src, CFL_SIZE);
	
	void* args[d->OO + d->II];
	for (uint j = 0; j < d->OO; j++)
		args[j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, src);
	for (uint j = 0; j < d->II; j++)
		args[d->OO + j] = d->inputs[j];
	nlop_generic_apply_with_opts_unchecked(d->nlop, d->OO + d->II, (void**)args, d->opts_no_der);
	for (uint j = 0; j < d->OO; j++)
		md_free(args[j]);

	int num_ops_par = 0;
	const struct operator_s* der_ops[d->OO];

	for (uint j = 0; j < d->OO; j++) {

		if (op_options_is_set_io(d->opts_no_der, j, i, OP_APP_NO_DER))
			continue;
		
		if( NULL == d->der_out[i + d->II * j])
			d->der_out[i + d->II * j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, src);
		
		der_ops[num_ops_par] = nlop_get_derivative(d->nlop, o, j)->forward;
		args[num_ops_par++] = d->der_out[i + d->II * j];
	}

	operator_apply_parallel_unchecked(num_ops_par, der_ops, (complex float**)args, src);

	nlop_clear_derivative(d->nlop);

	assert(NULL != d->der_out[i + d->II * o]);
	md_copy(d->DO[o], d->odims[o], dst, d->der_out[i + d->II * o], CFL_SIZE);
	if (d->der_once) {

		md_free(d->der_out[i + d->II * o]);
		d->der_out[i + d->II * o] = NULL;
	}
}


static void checkpoint_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);
	
	if ((NULL != d->adj_in[o]) && (1.e-8 > md_zrmse(d->DO[o], d->odims[o], d->adj_in[o], src))) {

		assert(NULL != d->adj_out[i + d->II * o]);
		md_copy(d->DI[i], d->idims[i], dst, d->adj_out[i + d->II * o], CFL_SIZE);
		if (d->der_once) {

			md_free(d->adj_out[i + d->II * o]);
			d->adj_out[i + d->II * o] = NULL;
		}
		return;
	}

	if (NULL == d->adj_in[o])
		d->adj_in[o] = md_alloc_sameplace(d->DO[o], d->odims[o], CFL_SIZE, src);
	md_copy(d->DO[o], d->odims[o], d->adj_in[o], src, CFL_SIZE);
	
	void* args[d->OO + d->II];
	for (uint j = 0; j < d->OO; j++)
		args[j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, src);
	for (uint j = 0; j < d->II; j++)
		args[d->OO + j] = d->inputs[j];
	nlop_generic_apply_with_opts_unchecked(d->nlop, d->OO + d->II, (void**)args, d->opts_no_der);
	for (uint j = 0; j < d->OO; j++)
		md_free(args[j]);

	int num_ops_par = 0;
	const struct operator_s* adj_ops[d->II];

	for (uint j = 0; j < d->II; j++) {

		if (op_options_is_set_io(d->opts_no_der, o, j, OP_APP_NO_DER))
			continue;
		
		if( NULL == d->adj_out[j + d->II * o])
			d->adj_out[j + d->II * o] = md_alloc_sameplace(d->DI[j], d->idims[j], CFL_SIZE, src);
		
		adj_ops[num_ops_par] = nlop_get_derivative(d->nlop, o, j)->adjoint;
		args[num_ops_par++] = d->adj_out[j + d->II * o];
	}

	operator_apply_parallel_unchecked(num_ops_par, adj_ops, (complex float**)args, src);

	nlop_clear_derivative(d->nlop);

	assert(NULL != d->adj_out[i + d->II * o]);
	md_copy(d->DI[i], d->idims[i], dst, d->adj_out[i + d->II * o], CFL_SIZE);
	if (d->der_once) {

		md_free(d->adj_out[i + d->II * o]);
		d->adj_out[i + d->II * o] = NULL;
	}
}




static void checkpoint_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);

	nlop_free(d->nlop);

	op_options_free(d->opts_no_der);

	for (uint i = 0; i < d->II; i++)
		xfree(d->idims[i]);
	
	for (uint i = 0; i < d->OO; i++)
		xfree(d->odims[i]);

	checkpoint_free_der(d);
	xfree(d->adj_in);
	xfree(d->adj_out);
	xfree(d->der_in);
	xfree(d->der_out);

	for (uint i = 0; i < d->II; i++) {

		md_free(d->inputs[i]);
		xfree(d->idims[i]);
	}
	xfree(d->idims);
	xfree(d->inputs);
	xfree(d->DI);

	for (uint i = 0; i < d->OO; i++)
		xfree(d->idims[i]);
	xfree(d->odims);
	xfree(d->DO);

	xfree(d);
}

static const char* nlop_graph_checkpointing(const nlop_data_t* _data, unsigned int N, unsigned int D[N], const char** arg_nodes[N], graph_t opts)
{
	auto data = CAST_DOWN(checkpoint_s, _data);

	return operator_graph_container(operator_get_graph_string(data->nlop->op, N, D, arg_nodes, opts), "checkpoint", _data, false);
}

const struct nlop_s* nlop_checkpoint_create(const struct nlop_s* nlop, bool der_once)
{

	PTR_ALLOC(struct checkpoint_s, d);
	SET_TYPEID(checkpoint_s, d);

	unsigned int II = nlop_get_nr_in_args(nlop);
	unsigned int OO = nlop_get_nr_out_args(nlop);

	unsigned int max_DI = 0;
	unsigned int max_DO = 0;

	PTR_ALLOC(unsigned int[OO], DO);
	PTR_ALLOC(const long*[OO], odims);
	PTR_ALLOC(unsigned int[II], DI);
	PTR_ALLOC(const long*[II], idims);
	
	for (uint i = 0; i < OO; i++) {
		
		auto iov = nlop_generic_codomain(nlop, i);
		(*DO)[i] = iov->N;
		max_DO = MAX(max_DO, iov->N);

		PTR_ALLOC(long[iov->N], tdims);
		md_copy_dims(iov->N, *tdims, iov->dims);
		(*odims)[i] = *PTR_PASS(tdims);
	}
	
	for (uint i = 0; i < II; i++) {
		
		auto iov = nlop_generic_domain(nlop, i);
		(*DI)[i] = iov->N;
		max_DI = MAX(max_DI, iov->N);

		PTR_ALLOC(long[iov->N], tdims);
		md_copy_dims(iov->N, *tdims, iov->dims);
		(*idims)[i] = *PTR_PASS(tdims);
	}

	d->DO = *PTR_PASS(DO);
	d->odims = * PTR_PASS(odims);
	d->DI = *PTR_PASS(DI);
	d->idims = * PTR_PASS(idims);


	d->nlop = nlop_clone(nlop);
	d->opts_no_der = NULL;

	d->der_once = der_once;
	
	d->II = II;
	d->OO = OO;

	PTR_ALLOC(complex float*[II], inputs);
	PTR_ALLOC(complex float*[II], der_in);
	PTR_ALLOC(complex float*[OO], adj_in);
	PTR_ALLOC(complex float*[II * OO], der_out);
	PTR_ALLOC(complex float*[II * OO], adj_out);

	d->inputs = *PTR_PASS(inputs);
	d->der_in = *PTR_PASS(der_in);
	d->der_out = *PTR_PASS(der_out);
	d->adj_in = *PTR_PASS(adj_in);
	d->adj_out = *PTR_PASS(adj_out);

	for (uint i = 0; i < II; i++)
		d->inputs[i] = NULL;
	for (uint i = 0; i < II; i++)
		d->der_in[i] = NULL;
	for (uint i = 0; i < OO; i++)
		d->adj_in[i] = NULL;
	for (uint i = 0; i < II * OO; i++)
		d->adj_out[i] = NULL;
	for (uint i = 0; i < II * OO; i++)
		d->der_out[i] = NULL;


	long nl_odims[OO][max_DO];
	long nl_idims[II][max_DI];
	
	for (uint i = 0; i < OO; i++){
		
		md_singleton_dims(max_DO, nl_odims[i]);
		md_copy_dims(d->DO[i], nl_odims[i], d->odims[i]);
	}

	for (uint i = 0; i < II; i++){
		
		md_singleton_dims(max_DI, nl_idims[i]);
		md_copy_dims(d->DI[i], nl_idims[i], d->idims[i]);
	}


	operator_property_flags_t props[II][OO];
	nlop_der_fun_t der_funs[II][OO];
	nlop_der_fun_t adj_funs[II][OO];

	for(uint i = 0; i < II; i++)
		for(uint o = 0; o < OO; o++) {

			props[i][o] = operator_get_property_io_flag(nlop->op, o, i);
			der_funs[i][o] = checkpoint_der;
			adj_funs[i][o] = checkpoint_adj;
		}

	const struct nlop_s* result = nlop_generic_with_props_create(	OO, max_DO, nl_odims, II, max_DI, nl_idims, CAST_UP(PTR_PASS(d)),
							checkpoint_fun, der_funs, adj_funs, NULL, NULL, checkpoint_del, NULL, props, nlop_graph_checkpointing);
	
	for (uint i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(nlop, i);
		result = nlop_reshape_in_F(result, i, iov->N, iov->dims);
	}
	for (uint o = 0; o < OO; o++) {

		auto iov = nlop_generic_codomain(nlop, o);
		result = nlop_reshape_out_F(result, o, iov->N, iov->dims);
	}

	return result;
}

const struct nlop_s* nlop_checkpoint_create_F(const struct nlop_s* nlop, bool der_once)
{
	auto result = nlop_checkpoint_create(nlop, der_once);
	nlop_free(nlop);
	return result;
}

