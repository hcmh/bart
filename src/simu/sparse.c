/*
 * Sparse matrix vector products
 *
 * Authors:
 * 2021 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */

#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "simu/sparse.h"
#include "linops/linop.h"
#include "simu/fd_geometry.h"
#include <math.h>
#include <complex.h>



/*
 * Allocate memory for sparse diagonal tensor
 *
 * @param N		number of dimensions
 * @param len		main diagonal length
 * @param N_diags	number of (off-)diagonals
 * @param offsets	offsets
 *
 * @returns 		Pointer to sparse tensor struct
 */
struct sparse_diag_s* sparse_diag_alloc(const long N, const long len, const long N_diags,
				const long (*offsets)[N])
{
	PTR_ALLOC(struct sparse_diag_s, mat);

	assert(N >= 1);
	assert(len >= 1);
	assert((N_diags >= 1) && (N_diags <= pow(2*len - 1, N - 1)));

	mat->N = N;
	mat->len = len;
	mat->N_diags = N_diags;

	// allocate
	mat->offsets = (long **)malloc(sizeof(long *) * N_diags + sizeof(long) * N * N_diags);
	for (int i = 0; i < N_diags; i++)
		mat->offsets[i] = ((long *)(mat->offsets + N_diags)) + i * N;
	mat->dims = (long *)malloc(N_diags * sizeof(long));

	long elements = 0;
	for (long i = 0; i < N_diags; i++) {
		long max_offset = -len, min_offset = len;
		for (long j = 0; j < N; j++) {
			long offset = offsets[i][j];
			mat->offsets[i][j] = offset;
			max_offset = MAX(offset, max_offset);
			min_offset = MIN(offset, min_offset);
		}
		assert((max_offset < len) && (min_offset > -len) && (max_offset - min_offset < len));

		// normalize offsets -> min_offset = 0
		long min_check = len;
		max_offset = 0;
		for (long j = 0; j < N; j++) {
			mat->offsets[i][j] -= min_offset;
			min_check =  MIN(mat->offsets[i][j], min_check);
			max_offset = MAX(mat->offsets[i][j], max_offset);
		}
		assert(min_check == 0);

		mat->dims[i] = len - max_offset;
		elements += mat->dims[i];
	}

	mat->offsets_normal = true;

	mat->diags = (complex float **)malloc(sizeof(complex float *) * N_diags + sizeof(complex float) * elements);
	long offset = 0;
	for (int i = 0; i < N_diags; i++) {
		mat->diags[i] = ((complex float *)(mat->diags + N_diags)) + offset;
		offset += mat->dims[i];
	}

	return PTR_PASS(mat);
}


void sparse_diag_free(struct sparse_diag_s *mat)
{
	xfree(mat->diags);
	xfree(mat->dims);
	xfree(mat->offsets);
	xfree(mat);
}

/*
 * Create a sparse tensor with constant entries along (off-)diagonals
 *
 * @param N		number of dimensions
 * @param len		main diagonal length
 * @param N_diags	number of (off-)diagonals
 * @param offsets	offsets
 * @param values	values on th (off-)diagonals
 */
struct sparse_diag_s * sparse_cdiags_create(const long N, const long len, const long N_diags,
				const long (*offsets)[N], const complex float values[N_diags])
{
	struct sparse_diag_s *mat = sparse_diag_alloc(N, len, N_diags, offsets);

	for (int i = 0; i < N_diags; i++) {
		// keep const qualifier
		md_zfill(1, mat->dims + i,  mat->diags[i], values[i]);
	}

	return mat;
}


void sparse_diag_to_dense(const long N, const long dims[N], complex float *out, const struct sparse_diag_s *mat)
{
	assert(N == mat->N);
	for (int i = 0; i < N; i++)
		assert(dims[i] == mat->len);

	long strs[N];
	md_calc_strides(N,strs, dims, CFL_SIZE);
	long diag_str = 0;
	for(int i = 0; i < N; i++)
		diag_str += strs[i];

	md_zfill(N, dims, out, 0);

	for (int i = 0; i < mat->N_diags; i++) {
		long offset = 0;

		for(int k = 0; k < N; k++)
			offset += mat->offsets[i][k]*strs[k];

		for(int j = 0; j < mat->dims[i]; j++)
			*((complex float *)( (void *)out + offset + j * diag_str)) = mat->diags[i][j];
	}
}



void calc_index_strides(const long N, long index_strides[N], const long dims[N])
{
	index_strides[0] = 1;
	for (int i = 1; i < N; i++)
		index_strides[i] = index_strides[i-1] * dims[i-1];
}


long calc_index_size(const long N, const long index_strides[N], const long dims[N])
{
	long size = 1;
	for (int i = 0; i < N; i++)
		size += (dims[i] - 1) * index_strides[i];
	return size;
}


long calc_index(const long N, const long index_strides[N], const long pos[N])
{
	long index = 0;
	for (int i = 0; i < N; i++)
		index += pos[i] * index_strides[i];
	return index;
}


static void clear_row(const long mat_index, struct sparse_diag_s *mat)
{
	assert(mat->N == 2);
	assert(mat_index < mat->len);
	for (long j = 0; j < mat->N_diags; j++) {
		const long *offsets = mat->offsets[j];
		// lower diagonals
		if ((offsets[0] > 0) && (mat_index >= offsets[0]))
			mat->diags[j][mat_index - offsets[0]] = 0;
		// upper and main diagonals
		if ((offsets[1] >= 0) && (mat_index < mat->dims[j]))
			mat->diags[j][mat_index] = 0;
	}
}



void sd_mask(const long N, const long dims[N], struct sparse_diag_s *mat, const complex float *mask)
{
	long pos[N], strs[N], index_strs[N];
	const long flags = (MD_BIT(N) - 1);

	md_set_dims(N, pos, 0);
	md_calc_strides(N, strs, dims, CFL_SIZE);
	calc_index_strides(N, index_strs, dims);

	do {
		long offset = md_calc_offset(N, strs, pos);
		long mat_index = calc_index(N, index_strs, pos);
		if ( creal(*((complex float *) ((void *)mask + offset))) < 1)
			clear_row(mat_index, mat);

	} while (md_next(N, dims, flags, pos));

}


void sd_matvec(long N, long dims[N], complex float *out, const complex float *vec, const struct sparse_diag_s *mat)
{
	assert(N == 1);
	assert(mat->len == dims[0]);
	md_zfill(N, dims, out, 0);

	for (int i = 0; i < mat->N_diags; i++)
		md_zfmac(1, &mat->dims[i], out + mat->offsets[i][0], mat->diags[i], vec + mat->offsets[i][1]);
}


static void sd_matTvec(long N, long dims[N], complex float *out, const complex float *vec, const struct sparse_diag_s *mat)
{
	assert(N == 1);
	assert(mat->len == dims[0]);
	md_zfill(N, dims, out, 0);

	for (int i = 0; i < mat->N_diags; i++)
		md_zfmacc(1, &mat->dims[i], out + mat->offsets[i][1], vec + mat->offsets[i][0], mat->diags[i]);
}

struct sd_matvec_s
{
	INTERFACE(linop_data_t);
	long *dims;
	long N;
	struct sparse_diag_s *mat;
	complex float *tmp;
};

static DEF_TYPEID(sd_matvec_s);



static void sd_matvec_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(sd_matvec_s, _data);

	sd_matvec(data->N, data->dims, dst, src, data->mat);
}



static void sd_matvec_adjoint_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(sd_matvec_s, _data);

	sd_matTvec(data->N, data->dims, dst, src, data->mat);
}



static void sd_matvec_normal_apply(const linop_data_t *_data, complex float *dst, const complex float *src)
{
	const auto data = CAST_DOWN(sd_matvec_s, _data);

	sd_matvec(data->N, data->dims, data->tmp, src, data->mat);
	sd_matTvec(data->N, data->dims, dst, data->tmp, data->mat);
}



static void sd_matvec_free(const linop_data_t *_data)
{
	const auto data = CAST_DOWN(sd_matvec_s, _data);
	xfree(data->dims);
	sparse_diag_free(data->mat);
	xfree(data->tmp);
}



struct linop_s *linop_sd_matvec_create(const long N, const long dims[N], struct sparse_diag_s* mat)
{
	PTR_ALLOC(struct sd_matvec_s, data);
	SET_TYPEID(sd_matvec_s, data);

	assert(N == 1);
	assert(dims[0] == mat->len);

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);
	data->N = N;
	//FIXME
	data->mat = mat;

	//FIXME
	data->tmp = md_alloc(N, dims, CFL_SIZE);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), sd_matvec_apply, sd_matvec_adjoint_apply, sd_matvec_normal_apply, NULL, sd_matvec_free);

}
