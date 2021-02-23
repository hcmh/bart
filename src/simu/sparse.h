#ifndef N_boundary_point_s
#define N_boundary_point_s 4
struct boundary_point_s {
	long index[N_boundary_point_s];
	_Complex float val;
	float dir[N_boundary_point_s];
};
#endif


struct sparse_diag_s {
	long N;
	long len;
	long N_diags;
	long **offsets;
	long *dims;
	float **diags;
	_Bool offsets_normal;
};

struct sparse_diag_s* sparse_diag_alloc(const long N, const long len, const long N_diags,
				const long (*offsets)[N]);

void sparse_diag_free(struct sparse_diag_s *);

struct sparse_diag_s * sparse_cdiags_create(const long N, const long len, const long N_diags,
				const long (*offsets)[N], const float values[N_diags]);

void sparse_diag_to_dense(const long N, const long dims[N], float *out, const struct sparse_diag_s *mat);


long calc_index_strides(const long N, long index_strides[N], const long dims[N]);
long calc_index_size(const long N, const long index_strides[N], const long dims[N]);
long calc_index(const long N, const long index_strides[N], const long pos[N]);

struct sparse_diag_s* sd_laplace_create(long N, const long dims[N]);

void laplace_dirichlet(struct sparse_diag_s *mat, const long N, const long dims[N], const long n_points, struct boundary_point_s points[n_points]);

void sd_matvec(long N, long dims[N], float *out, float *vec, const struct sparse_diag_s *mat);
