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

	mat->diags = (float **)malloc(sizeof(float *) * N_diags + sizeof(float) * elements);
	long offset = 0;
	for (int i = 0; i < N_diags; i++) {
		mat->diags[i] = ((float *)(mat->diags + N_diags)) + offset;
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


struct sparse_diag_s* sd_laplace_create(long N, const long dims[N])
{
	long len = 1, N_mat = 2, N_diags = N*2 + 1, offsets[N_diags][N_mat];
	float values[N_diags];

	//main diagonal
	offsets[0][0] = 0;
	offsets[0][1] = 0;
	values [0]    = 2 * N;

	//off diagonals
	int n = 1;
	for (int i = 0; i < N; i++) {
		offsets[n][0] = 0;
		offsets[n][1] = len;
		values [n++]  = -1;

		offsets[n][0] = 0;
		offsets[n][1] = -len;
		values [n++]  = -1;

		len *= dims[i];
	}

	struct sparse_diag_s* mat = sparse_cdiags_create(N_mat, len, N_diags, offsets, values);

	//correct boundaries
	n = 1;
	long str = 1;
	long len_ones, len_zeros;
	for (int i = 0; i < N; i++) {
		len_ones  = (dims[i] - 1)*str;
		len_zeros = str;
		long j = len_ones;
		while (j + len_zeros - 1 < mat->dims[n]) {
			for (long k = 0; k < len_zeros; k++) {
				mat->diags[n    ][j + k] = 0;
				mat->diags[n + 1][j + k] = 0;
			}
			j += len_zeros + len_ones;
		}
		n += 2;

		str *= dims[i];
	}

	return mat;
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

void laplace_dirichlet(struct sparse_diag_s *mat, const long N, const long dims[N], const long n_points, struct boundary_point_s points[n_points])
{
	assert(2 == mat->N);

	long str[N];
	calc_index_strides(N, str, dims);
	assert(mat->len == calc_index_size(N, str, dims));

	assert(mat->offsets_normal);

	//for each point on a boundary
	for (long i = 0; i < n_points; i++) {
		long mat_index = calc_index(N, str, points[i].index);

		// set main diagonal to 1
		mat->diags[0][mat_index] = 1;

		// set off-diagonals to 0
		for (long j = 1; j < mat->N_diags; j++) {
			const long *offsets = mat->offsets[j];
			// lower diagonals
			if ((offsets[0] > 0) && (mat_index >= offsets[0]))
				mat->diags[j][mat_index - offsets[0]] = 0;
			// upper diagonals
			if ((offsets[1] > 0) && (mat_index < mat->dims[j]))
				mat->diags[j][mat_index] = 0;
		}
	}
}



void laplace_neumann(struct sparse_diag_s *mat, const long N, const long dims[N], const long n_points, struct boundary_point_s boundary[n_points])
{
	assert(2 == mat->N);

	long str[N];
	calc_index_strides(N, str, dims);
	assert(mat->len == calc_index_size(N, str, dims));

	assert(mat->offsets_normal);

	for (long i = 0; i < n_points; i++) {
		struct boundary_point_s *point = boundary + i;
		long mat_index = calc_index(N, str, point->index);
		long diag_index = 0;
		long  ext_neighbours = 0;
		for (long j = 0; j < N; j++)
			ext_neighbours += abs(point->dir[j]);

		mat->diags[diag_index][mat_index] -= ext_neighbours;

		// set this row in the off-diagonals
		for (long j = 0; j < N; j++) {
			// upper diagonal
			const long *offsets = mat->offsets[++diag_index];
			assert(offsets[0] == 0);
			if (point->dir[j] == 1)
				mat->diags[diag_index][mat_index] = 0;

			// lower diagonal
			offsets = mat->offsets[++diag_index];
			assert(offsets[0] > 0);
			if (point->dir[j] == -1)
				mat->diags[diag_index][mat_index - offsets[0]] = 0;

		}
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


void sd_matvec(long N, long dims[N], float *out, float *vec, const struct sparse_diag_s *mat)
{
	assert(N == 1);
	assert(mat->len == dims[0]);
	md_clear(N, dims, out, sizeof(float));

	for (int i = 0; i < mat->N_diags; i++)
		md_fmac(1, &mat->dims[i], out + mat->offsets[i][0], mat->diags[i], vec + mat->offsets[i][1]);
}
