/*
 * Sparse tensor operations
 *
 * Authors:
 * 2021 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 */

#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "simu/sparse.h"
#include <math.h>



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
	mat->offsets = (long **)malloc(sizeof(long *) * N_diags + sizeof(long) * N);
	for (int i = 0; i < N_diags; i++)
		mat->offsets[i] = ((long *)(mat->offsets + N_diags)) + i * N;
	mat->dims = (long *)calloc(N_diags, sizeof(long));

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

	mat->diags = malloc(sizeof(float *) * N_diags + sizeof(float) * elements);
	long offset = 0;
	for (int i = 0; i < N_diags; i++) {
		mat->diags[i] = ((float *)(mat->diags + N_diags)) + offset;
		offset += mat->dims[i];
	}

	return PTR_PASS(mat);
}


void sparse_diag_free(struct sparse_diag_s *mat)
{
	free(mat->diags);
	free(mat->dims);
	free(mat->offsets);
	free(mat);
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
				const long (*offsets)[N], const float values[N_diags])
{
	struct sparse_diag_s *mat = sparse_diag_alloc(N, len, N_diags, offsets);

	for (int i = 0; i < N_diags; i++) {
		// keep const qualifier
		float a = values[i];
		md_fill(1, mat->dims + i,  mat->diags[i], &a, sizeof(float));
	}

	return mat;
}


void sparse_diag_to_dense(const long N, const long dims[N], float *out, const struct sparse_diag_s *mat)
{
	assert(N == mat->N);
	for (int i = 0; i < N; i++)
		assert(dims[i] == mat->len);

	long strs[N];
	md_calc_strides(N,strs, dims, sizeof(float));
	long diag_str = 0;
	for(int i = 0; i < N; i++)
		diag_str += strs[i];

	float null = 0;
	md_fill(N, dims, out, &null, sizeof(float));

	for (int i = 0; i < mat->N_diags; i++) {
		long offset = 0;

		for(int k = 0; k < N; k++)
			offset += mat->offsets[i][k]*strs[k];

		for(int j = 0; j < mat->dims[i]; j++)
			*((float *)( (void *)out + offset + j * diag_str)) = mat->diags[i][j];
	}
}
