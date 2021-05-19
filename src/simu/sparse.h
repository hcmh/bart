struct sparse_diag_s {
	long N;
	long len;
	long N_diags;
	long **offsets;
	long *dims;
	_Complex float **diags;
	_Bool offsets_normal;
};

struct sparse_diag_s *sparse_diag_alloc(const long N, const long len, const long N_diags,
					const long (*offsets)[N]);

void sparse_diag_free(struct sparse_diag_s *);

struct sparse_diag_s *sparse_cdiags_create(const long N, const long len, const long N_diags,
					   const long (*offsets)[N], const _Complex float values[N_diags]);

void sparse_diag_to_dense(const long N, const long dims[N], _Complex float *out, const struct sparse_diag_s *mat);


void calc_index_strides(const long N, long index_strides[N], const long dims[N]);
long calc_index_size(const long N, const long index_strides[N], const long dims[N]);
long calc_index(const long N, const long index_strides[N], const long pos[N]);

void sd_matvec(long N, long dims[N], _Complex float *out, const _Complex float *vec, const struct sparse_diag_s *mat);

void sd_mask(const long N, const long dims[N], struct sparse_diag_s *mat, const _Complex float *mask);

struct linop_s *linop_sd_matvec_create(const long N, const long dims[N], struct sparse_diag_s *mat);
