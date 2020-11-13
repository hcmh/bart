#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "num/multind.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/chain.h"

#include "nn/nn.h"
#include "nn/init.h"

#include "weights.h"



nn_weights_t nn_weights_create(int N, const struct iovec_s* iovs[N])
{
	PTR_ALLOC(struct nn_weights_s, result);
	result->N = N;

	PTR_ALLOC(const struct iovec_s*[N], niov);
	PTR_ALLOC(_Complex float*[N], ntensors);

	for (int i = 0; i < N; i++) {

		(*niov)[i] = iovec_create2(iovs[i]->N, iovs[i]->dims, iovs[i]->strs, iovs[i]->size);
		(*ntensors)[i] = md_alloc(iovs[i]->N, iovs[i]->dims, iovs[i]->size);
	}

	result->iovs = *PTR_PASS(niov);
	result->tensors = *PTR_PASS(ntensors);
	return PTR_PASS(result);
}

nn_weights_t load_nn_weights(const char *name)
{
	int N_max = 64;
	unsigned int D_max = 64;
	unsigned int D[N_max];
	long dimensions[N_max][D_max];
	complex float* args[N_max];

	unsigned int N = load_multi_cfl(name, N_max, D_max, D, dimensions, args);

	PTR_ALLOC(struct nn_weights_s, result);
	result->N = N;

	PTR_ALLOC(const struct iovec_s*[N], niov);
	PTR_ALLOC(_Complex float*[N], ntensors);

	const long* dimensions_unmap[N];

	for (unsigned int i = 0; i < N; i++) {

		(*niov)[i] = iovec_create(D[i], dimensions[i], sizeof(_Complex float));
		(*ntensors)[i] = md_alloc(D[i], dimensions[i], sizeof(_Complex float));
		md_copy(D[i], dimensions[i], (*ntensors)[i], args[i], sizeof(_Complex float));

		dimensions_unmap[i] = dimensions[i];
	}

	result->iovs = *PTR_PASS(niov);
	result->tensors = *PTR_PASS(ntensors);

	unmap_multi_cfl(result->N, D, dimensions_unmap, args);

	return PTR_PASS(result);
}

void dump_nn_weights(const char *name, nn_weights_t weights) {

	unsigned int D[weights->N];
	const long* dims[weights->N];
	for (int i = 0; i < weights->N; i++) {

		D[i] = weights->iovs[i]->N;
		dims[i] = weights->iovs[i]->dims;
	}

	dump_multi_cfl(name, weights->N, D, dims, (const complex float**)weights->tensors);
}

void move_gpu_nn_weights(nn_weights_t weights){
#ifdef USE_CUDA
	for (int i = 0; i < weights->N; i++) {

		auto iov = weights->iovs[i];
		complex float* tmp = md_alloc_gpu(iov->N, iov->dims, iov->size);
		md_copy(iov->N, iov->dims, tmp, weights->tensors[i], iov->size);
		md_free(weights->tensors[i]);
		weights->tensors[i] = tmp;
	}
#else
	error("Compiled without gpu support!");
	UNUSED(weights);
#endif
}

bool nn_weights_on_gpu(nn_weights_t weights)
{
#ifdef USE_CUDA
	return cuda_ondevice(weights->tensors[0]);
#else
	UNUSED(weights);
	return false;
#endif
}

void nn_weights_free(nn_weights_t weights){

	for (int i = 0; i < weights->N; i++) {

		iovec_free(weights->iovs[i]);
		md_free(weights->tensors[i]);
	}

	xfree(weights->tensors);
	xfree(weights->iovs);

	xfree(weights);
}

void nn_init(nn_t op, nn_weights_t weights)
{
	for (uint i = 0, ip = 0; i < nn_get_nr_in_args(op); i++){

		if(NULL != op->initializers[i]) {

			auto iov = nlop_generic_domain(op->network, i);
			iovec_check(weights->iovs[ip], iov->N, iov->dims, iov->strs);
			initializer_apply(op->initializers[i], iov->N, iov->dims, weights->tensors[ip++]);
		}
	}
}

nn_weights_t nn_weights_create_from_nn(nn_t x)
{
	int N = nn_get_nr_weights(x);
	const struct iovec_s* iovs[N];

	for (int i = 0, ip = 0; i < nlop_get_nr_in_args(x->network); i++)
		if (NULL != x->initializers[i])
			iovs[ip++] = nlop_generic_domain(x->network, i);

	return nn_weights_create(N, iovs);
}


const struct nlop_s* nn_get_nlop_wo_weights(nn_t op, nn_weights_t weights, bool copy)
{
	assert(weights->N == nn_get_nr_weights(op));

	auto result = nlop_clone(op->network);

	for (int i = (int)nn_get_nr_out_args(op) - 1; i >= 0; i--)
		if (OUT_BATCHNORM == op->out_types[i])
			result = nlop_del_out_F(result, i);

	for (int i = (int)nn_get_nr_in_args(op) - 1, ip = weights->N - 1; i >= 0; i--)
		if ((IN_OPTIMIZE == op->in_types[i]) || (IN_BATCHNORM == op->in_types[i])) {

			auto iov = weights->iovs[ip];
			result = nlop_set_input_const_F(result, i, iov->N, iov->dims, copy, weights->tensors[ip--]);
		}

	return result;
}

const struct nlop_s* nn_get_nlop_wo_weights_F(nn_t op, nn_weights_t weights, bool copy)
{
	auto result = nn_get_nlop_wo_weights(op, weights, copy);
	nn_free(op);

	return result;
}

void nn_weights_copy(nn_weights_t dst, nn_weights_t src){

	assert(dst->N == src->N);
	for(int i = 0; i < src->N; i++){

		auto iovd = dst->iovs[i];
		auto iovs = src->iovs[i];

		assert(iovd->N == iovs->N);
		assert(iovd->size == iovs->size);
		for (uint j = 0; j < iovd->N; j++)
			assert((1 == iovs->dims[j] ) || (iovs->dims[j] == iovs->dims[j]));

		md_copy2(iovd->N, iovd->dims,
			MD_STRIDES(iovd->N, iovd->dims, iovd->size), dst->tensors[i],
			MD_STRIDES(iovs->N, iovs->dims, iovs->size), src->tensors[i],
			iovs->size);
	}
}



const struct nlop_s* deflatten_weights_create(const struct nlop_s* network, unsigned long flag)
{
	int count = 0;
	long size_in = 0;
	for (int i = 0; i < nlop_get_nr_in_args(network); i++)
		if(!(MD_IS_SET(flag, i))){

			count += 1;
			size_in += md_calc_size(nlop_generic_domain(network, i)->N, nlop_generic_domain(network, i)->dims);
	}

	assert(0 < count);

	struct nlop_s* result = NULL;
	long pos = 0;

	for(int i = 0, j = 0; i < count; i++){

		while(MD_IS_SET(flag, j))
			j += 1;

		const struct linop_s* lin_tmp = linop_copy_selected_create2(nlop_generic_domain(network, j)->N, nlop_generic_domain(network, j)->dims, nlop_generic_domain(network, j)->strs, size_in, pos);

		pos += md_calc_size(nlop_generic_domain(network, j)->N, nlop_generic_domain(network, j)->dims);

		if(result == NULL)
			result = nlop_from_linop_F(lin_tmp);
		else
			result = nlop_dup_F(nlop_combine_FF(result, nlop_from_linop_F(lin_tmp)), 0, 1);

		j += 1;

	}

	return result;
}

const struct nlop_s* deflatten_weights(const struct nlop_s* network, unsigned long flag)
{
	const struct nlop_s* deflatten = deflatten_weights_create(network, flag);
	const struct nlop_s* result = nlop_combine(network, deflatten);

	int o = nlop_get_nr_out_args(network);
	int count = nlop_get_nr_out_args(deflatten);

	nlop_free(deflatten);

	for(int i = 0, j = 0; i < count; i++){

		while(MD_IS_SET(flag, j))
			j += 1;
		
		result = nlop_link_F(result, o, j);
	}

	return result;
}

const struct nlop_s* deflatten_weightsF(const struct nlop_s* network, unsigned long flag)
{
	const struct nlop_s* result = deflatten_weights(network, flag);
	nlop_free(network);

	return result;
}
