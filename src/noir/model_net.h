
struct noir2_conf_s;

extern const struct nlop_s* noir_cart_unrolled_create(	int N,
							const long pat_dims[N], const _Complex float* pattern,
							const long bas_dims[N], const _Complex float* basis,
							const long msk_dims[N], const _Complex float* mask,
							const long ksp_dims[N],
							const long cim_dims[N],
							const long img_dims[N],
							const long col_dims[N],
							struct noir2_conf_s* conf);