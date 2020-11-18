#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "iter/iter6.h"

#include "batch_gen.h"



struct batch_gen_rand_s {

	INTERFACE(nlop_data_t);

	long D; //number of arrays

	long N;
	const long** dims;

	long Nb;
	long Nt;
	const complex float** data;

	unsigned int rand_seed;
};

DEF_TYPEID(batch_gen_rand_s);


static void batch_gen_rand_fun(const struct nlop_data_s* _data, int N, complex float* args[N])
{
	START_TIMER;
	const auto data = CAST_DOWN(batch_gen_rand_s, _data);

	long indices[data->Nb];
	for (int i = 0; i < data->Nb; i++)
		#pragma omp critical
		indices[i] = rand_r(&(data->rand_seed)) % data->Nt;

	assert(data->D == N);

	for (long j = 0; j < data->D; j++){

		if (1 == data->dims[j][data->N-1]){ //not batched data

			md_copy(data->N, data->dims[j], args[j], data->data[j], CFL_SIZE);
			continue;
		}

		size_t size_dataset =  md_calc_size(data->N-1, data->dims[j]);
		for (long i = 0; i < data->Nb; i++){

			long offset_src = indices[i] * size_dataset;
			long offset_dst = i * size_dataset;
			md_copy(data->N - 1, data->dims[j], args[j] + offset_dst, (complex float*)data->data[j] + offset_src, CFL_SIZE);
		}
	}
	PRINT_TIMER("frw batchgen rand");
}

static void batch_gen_rand_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(batch_gen_rand_s, _data);

	for(long i = 0; i < data->D; i ++)
		xfree(data->dims[i]);
	xfree(data->dims);
	xfree(data->data);
	xfree(data);
}

static const struct nlop_s* batch_gen_rand_with_seed_create(long D, long N, const long* dims[D], const complex float* data[D], long Nt, long Nc, unsigned int seed)
{
	PTR_ALLOC(struct batch_gen_rand_s, d);
	SET_TYPEID(batch_gen_rand_s, d);

	d->D = D;
	d->N = N;
	d->Nb = 0; //(dims[0][N-1]);
	d->Nt = Nt;
	PTR_ALLOC(const long*[D], ndims);
	PTR_ALLOC(const complex float*[D], ndata);

	for (long j = 0; j < D; j++) {

		PTR_ALLOC(long[N], dimsj);
		md_copy_dims(N, *dimsj, dims[j]);
		(*ndims)[j] = *PTR_PASS(dimsj);
		assert(	(0 == d->Nb) //d-Nt not set
			|| (d->Nb == dims[j][N-1]) // last dim equals
			|| (1 == dims[j][N-1]) // not batched
			);
		if (1 != dims[j][N-1])
			d->Nb = dims[j][N-1];

		(*ndata)[j] = data[j];
	}


	d->data = *PTR_PASS(ndata);
	d->dims = *PTR_PASS(ndims);

	d->rand_seed = seed;

	for (int i = 0; i < Nc; i++)
		for (int j = 0; j < d->Nb; j++)
			#pragma omp critical
			rand_r(&(d->rand_seed));


	assert(d->Nb <= d->Nt);

	long nl_odims[D][N];
	for(long i = 0; i < D; i++)
		md_copy_dims(N, nl_odims[i], dims[i]);

	return nlop_generic_create(D, N, nl_odims, 0, 0, NULL, CAST_UP(PTR_PASS(d)), batch_gen_rand_fun, NULL, NULL, NULL, NULL, batch_gen_rand_del);
}

/**
 * Create an operator copying Nb random (not necessarily distinct) datasets to the output
 *
 * @param D number of tensores
 * @param N number of dimensions
 * @param dims pointers to dims (batchsize or 1 in last dimension)
 * @param data pointers to data
 * @param Nt total number of available datasets
 * @param Nc number of calls (initializes the nlop as it had been applied Nc times) -> reproducible warm start
 */
const struct nlop_s* batch_gen_rand_create(long D, long N, const long* dims[D], const complex float* data[D], long Nt, long Nc)
{
	return batch_gen_rand_with_seed_create(D, N, dims, data, Nt, Nc, 123);
}

static void rand_perm_data(unsigned int* rand_seed, long N, long perm[N])
{
	bool drawn[N];
	for (int i = 0; i < N; i++)
		drawn[i] = false;

	for (int i = 0; i < N; i++) {

		#pragma omp critical
		perm[i] = rand_r(rand_seed) % (N - i);

		for (int j = 0; j < N; j++)
			if (drawn[j] && perm[i] >= j)
				perm[i] ++;

		drawn[perm[i]] = true;
	}
}

static void rand_perm_batches(unsigned int* rand_seed, long N, long perm[N], long Nb)
{
	long perm_batch[N / Nb];

	rand_perm_data(rand_seed, N / Nb, perm_batch);

	for (int i = 0; i < N / Nb; i++)
		for (int j = 0; j < Nb; j++)
			perm[Nb * i + j] = perm_batch[i] * Nb + j;

	for (int i = (N / Nb) * Nb; i < N; i++)
		perm[i] = i;
}

static void get_perm_batches(enum BATCH_GEN_TYPE type, unsigned int* rand_seed, long Nt, long perm[Nt], long Nb)
{
	switch (type) {

		case BATCH_GEN_SAME:

			for (int i = 0; i < Nt; i++)
				perm[i] = i;
			break;

		case BATCH_GEN_SHUFFLE_BATCHES:

			rand_perm_batches(rand_seed, Nt, perm, Nb);
			break;

		case BATCH_GEN_SHUFFLE_DATA:

			rand_perm_data(rand_seed, Nt, perm);
			break;

		case BATCH_GEN_RANDOM_DATA:

			assert(0);
			break;
	}
}



struct batch_gen_data_s {

	INTERFACE(nlop_data_t);

	long D; //number of arrays

	long N;
	const long** dims;

	long Nb;
	long Nt;
	const complex float** data;

	long start;

	long* perm;

	enum BATCH_GEN_TYPE type;

	unsigned int rand_seed;
};

DEF_TYPEID(batch_gen_data_s);


static void batch_gen_fun(const struct nlop_data_s* _data, int N, complex float* args[N])
{
	START_TIMER;
	const auto data = CAST_DOWN(batch_gen_data_s, _data);

	assert(data->D == N);
	long start = data->start;

	if (start + data->Nb > data->Nt) {

		start = 0;
		get_perm_batches(data->type, &(data->rand_seed), data->Nt, data->perm, data->Nb);
	}

	for (long j = 0; j < data->D; j++){

		if (1 == data->dims[j][data->N-1]){ //not batched data

			md_copy(data->N, data->dims[j], args[j], data->data[j], CFL_SIZE);
			continue;
		}

		size_t size_dataset =  md_calc_size(data->N-1, data->dims[j]);
		for (long i = 0; i < data->Nb; i++){

			long offset_src = (data->perm[(start + i)] % data->Nt) * size_dataset;
			long offset_dst = i * size_dataset;
			md_copy(data->N - 1, data->dims[j], args[j] + offset_dst, (complex float*)data->data[j] + offset_src, CFL_SIZE);
		}
	}

	data->start = (start + data->Nb);
	PRINT_TIMER("frw batchgen");
}

static void batch_gen_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(batch_gen_data_s, _data);

	for(long i = 0; i < data->D; i ++)
		xfree(data->dims[i]);
	xfree(data->dims);
	xfree(data->data);
	xfree(data->perm);
	xfree(data);
}



/**
 * Create an operator copying Nb random (not necessarily distinct) datasets to the output
 *
 * @param D number of tensores
 * @param N number of dimensions
 * @param dims pointers to dims (batchsize or 1 in last dimension)
 * @param data pointers to data
 * @param Nt total number of available datasets
 * @param Nc number of calls (initializes the nlop as it had been applied Nc times) -> reproducible warm start
 * @param type methode to compose new batches
 * @param seed seed for random reshuffeling of batches
 */
const struct nlop_s* batch_gen_create(long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc, enum BATCH_GEN_TYPE type, unsigned int seed)
{
	if (BATCH_GEN_RANDOM_DATA == type)
		return batch_gen_rand_with_seed_create(D, N, dims, data, Nt, Nc, seed);

	PTR_ALLOC(struct batch_gen_data_s, d);
	SET_TYPEID(batch_gen_data_s, d);

	d->D = D;
	d->N = N;
	d->Nb = 0; //(dims[0][N-1]);
	d->Nt = Nt;
	PTR_ALLOC(const long*[D], ndims);
	PTR_ALLOC(const complex float*[D], ndata);

	for (long j = 0; j < D; j++) {

		PTR_ALLOC(long[N], dimsj);
		md_copy_dims(N, *dimsj, dims[j]);
		(*ndims)[j] = *PTR_PASS(dimsj);
		assert(	(0 == d->Nb) //d-Nt not set
			|| (d->Nb == dims[j][N-1]) // last dim equals
			|| (1 == dims[j][N-1]) // not batched
			);
		if (1 != dims[j][N-1])
			d->Nb = dims[j][N-1];

		(*ndata)[j] = data[j];
	}


	d->data = *PTR_PASS(ndata);
	d->dims = *PTR_PASS(ndims);

	d->rand_seed = seed;
	d->type = type;

	PTR_ALLOC(long[Nt], perm);
	d->perm = *PTR_PASS(perm);
	d->start = 0;

	get_perm_batches(d->type, &(d->rand_seed), d->Nt, d->perm, d->Nb);

	for (int i = 0; i < Nc; i++) { //initializing the state after Nc calls to batchnorm

		if (d->start + d->Nb > d->Nt) {

			d->start = 0;
			get_perm_batches(d->type, &(d->rand_seed), d->Nt, d->perm, d->Nb);
		}
		d->start = (d->start + d->Nb);
	}

	assert(d->Nb <= d->Nt);

	long nl_odims[D][N];
	for(long i = 0; i < D; i++)
		md_copy_dims(N, nl_odims[i], dims[i]);

	return nlop_generic_create(D, N, nl_odims, 0, 0, NULL, CAST_UP(PTR_PASS(d)), batch_gen_fun, NULL, NULL, NULL, NULL, batch_gen_del);
}

/**
 * Create operator copying Nb datasets to the output
 *
 * @param D number of tensores
 * @param N number of dimensions
 * @param dims pointers to dims (batchsize or 1 in last dimension)
 * @param data pointers to data
 * @param Nt total number of available datasets
 * @param Nc number of calls (initializes the nlop as it had been applied Nc times) -> reproducible warm start
 */
const struct nlop_s* batch_gen_linear_create(long D, long N, const long* dims[D], const complex float* data[D], long Nt, long Nc)
{
	return batch_gen_create(D, N, dims, data, Nt, Nc, BATCH_GEN_SAME, 0);
}

const struct nlop_s* batch_gen_create_from_iter(struct iter6_conf_s* iter_conf, long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc)
{
	return batch_gen_create(D, N, dims, data, Nt, Nc, iter_conf->batchgen_type, iter_conf->batch_seed);
}