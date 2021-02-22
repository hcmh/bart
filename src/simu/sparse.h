
struct sparse_diag_s {
	long N;
	long len;
	long N_diags;
	long **offsets;
	long *dims;
	float **diags;
};

struct sparse_diag_s* sparse_diag_alloc(const long N, const long len, const long N_diags,
				const long (*offsets)[N]);

void sparse_diag_free(struct sparse_diag_s *);

struct sparse_diag_s * sparse_cdiags_create(const long N, const long len, const long N_diags,
				const long (*offsets)[N], const float values[N_diags]);

void sparse_diag_to_dense(const long N, const long dims[N], float *out, const struct sparse_diag_s *mat);
