struct config_nlop_mri_s {

	unsigned long coil_flags;
	unsigned long image_flags;
	unsigned long pattern_flags;
	unsigned long batch_flags;
	unsigned long fft_flags;
	unsigned long coil_image_flags;

	_Bool basis;
	_Bool gridded;
	_Bool noncart;

	struct nufft_conf_s* nufft_conf;
};

struct iter_conjgrad_conf;

extern struct config_nlop_mri_s conf_nlop_mri_simple;

extern const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);

extern const struct nlop_s* nlop_mri_normal_create(int N, const long cim_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_normal_inv_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);
extern const struct nlop_s* nlop_mri_dc_prox_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);
extern const struct nlop_s* nlop_mri_dc_pnorm_IRLS_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf, float p, int iter);

extern const struct nlop_s* nlop_mri_normal_max_eigen_create(int N, const long cim_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_scale_rss_create(int N, const long max_dims[N], const struct config_nlop_mri_s* conf);