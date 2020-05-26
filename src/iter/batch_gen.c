#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"

#include "batch_gen.h"


struct batch_gen_linear_s {

	INTERFACE(nlop_data_t);
	
	long D; //number of arrays

	long N;
	const long** dims;

	long Nb;
	long Nt;
	const complex float** data;
	long start;
};

DEF_TYPEID(batch_gen_linear_s);


static void batch_gen_linear_fun(const struct nlop_data_s* _data, int N, complex float* args[N])
{
	START_TIMER;
	const auto data = CAST_DOWN(batch_gen_linear_s, _data);

	assert(data->D == N);
	long start = data->start;
	
	if (start + data->Nb > data->Nt)				// we ignore the last datsets if they do not fit in a batch
		start = start % ((data->Nt / data->Nb) * data->Nb);	// -> the batches have the same content in each epoch
	
	for (long j = 0; j < data->D; j++){

		if (1 == data->dims[j][data->N-1]){ //not batched data

			md_copy(data->N, data->dims[j], args[j], data->data[j], CFL_SIZE);
			continue;
		}
			
		size_t size_dataset =  md_calc_size(data->N-1, data->dims[j]);
		for (long i = 0; i < data->Nb; i++){

			long offset_src = ((start + i) % data->Nt) * size_dataset;
			long offset_dst = i * size_dataset;
			md_copy(data->N - 1, data->dims[j], args[j] + offset_dst, (complex float*)data->data[j] + offset_src, CFL_SIZE);
		}
	}

	data->start = (start + data->Nb);
	PRINT_TIMER("frw batchgen");
}

static void batch_gen_linear_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(batch_gen_linear_s, _data);

	for(long i = 0; i < data->D; i ++)
		xfree(data->dims[i]);
	xfree(data->dims);
	xfree(data->data);
	xfree(data);
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
	PTR_ALLOC(struct batch_gen_linear_s, d);
	SET_TYPEID(batch_gen_linear_s, d);

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

	d->start = Nc * (d->Nb);

	d->data = *PTR_PASS(ndata);
	d->dims = *PTR_PASS(ndims);

	assert(d->Nb <= d->Nt);

	long nl_odims[D][N];
	for(long i = 0; i < D; i++)
		md_copy_dims(N, nl_odims[i], dims[i]);
	
	return nlop_generic_create(D, N, nl_odims, 0, 0, NULL, CAST_UP(PTR_PASS(d)), batch_gen_linear_fun, NULL, NULL, NULL, NULL, batch_gen_linear_del);
}


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

	d->rand_seed = 123;

	for (int i = 0; i < Nc * d->Nb; i++)
		#pragma omp critical
		rand_r(&(d->rand_seed));

	assert(d->Nb <= d->Nt);

	long nl_odims[D][N];
	for(long i = 0; i < D; i++)
		md_copy_dims(N, nl_odims[i], dims[i]);
	
	return nlop_generic_create(D, N, nl_odims, 0, 0, NULL, CAST_UP(PTR_PASS(d)), batch_gen_rand_fun, NULL, NULL, NULL, NULL, batch_gen_rand_del);
}