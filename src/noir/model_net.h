
struct noir2_conf_s;
struct noir2_s;
struct iter_conjgrad_conf;

extern int noir_model_get_N(struct noir2_s* model);
extern void noir_model_get_img_dims(int N, long img_dims[N], struct noir2_s* model);
extern void noir_model_get_col_dims(int N, long col_dims[N], struct noir2_s* model);
extern void noir_model_get_cim_dims(int N, long cim_dims[N], struct noir2_s* model);
extern long noir_model_get_size(struct noir2_s* model);
extern long noir_model_get_skip(struct noir2_s* model);

extern const struct nlop_s* noir_decomp_create(struct noir2_s* model);
extern const struct nlop_s* noir_decomp_batch_create(int Nb, struct noir2_s* model[Nb]);
extern const struct nlop_s* noir_split_create(struct noir2_s* model);
extern const struct nlop_s* noir_split_batch_create(int Nb, struct noir2_s* model[Nb]);
extern const struct nlop_s* noir_join_create(struct noir2_s* model);
extern const struct nlop_s* noir_join_batch_create(int Nb, struct noir2_s* model[Nb]);
extern const struct nlop_s* noir_cim_batch_create(int Nb, struct noir2_s* model[Nb]);
extern const struct nlop_s* noir_extract_img_batch_create(int Nb, struct noir2_s* model[Nb]);
extern const struct nlop_s* noir_set_img_batch_create(int Nb, struct noir2_s* model[Nb]);

extern const struct nlop_s* noir_gauss_newton_step_batch_create(int Nb, struct noir2_s* model[Nb], const struct iter_conjgrad_conf* iter_conf, float update, _Bool fix_coils);
extern const struct nlop_s* noir_adjoint_fft_create(struct noir2_s* model);
extern const struct nlop_s* noir_adjoint_fft_batch_create(int Nb, struct noir2_s* model[Nb]);

extern const struct nlop_s* noir_fft_create(struct noir2_s* model);
extern const struct nlop_s* noir_fft_batch_create(int Nb, struct noir2_s* model[Nb]);




#if 0
extern const struct nlop_s* noir_cart_unrolled_create(	int N,
							const long pat_dims[N], const _Complex float* pattern,
							const long bas_dims[N], const _Complex float* basis,
							const long msk_dims[N], const _Complex float* mask,
							const long ksp_dims[N],
							const long cim_dims[N],
							const long img_dims[N],
							const long col_dims[N],
							struct noir2_conf_s* conf);
#endif