struct config_nlop_mri_s {

	unsigned long coil_flags;
	unsigned long image_flags;
	unsigned long mask_flags;
	unsigned long pattern_flags;
	unsigned long batch_flags;
	unsigned long fft_flags;

	_Bool regrid;

	_Bool keep_mask_input;
	_Bool keep_lambda_input;

	float lambda_fixed;
	float lambda_init;
	struct iter_conjgrad_conf* iter_conf;
};

extern struct config_nlop_mri_s conf_nlop_mri_simple;

extern struct config_nlop_mri_s* config_nlop_mri_create(void);
extern void config_nlop_mri_free(const struct config_nlop_mri_s* config);

extern const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_normal_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);

extern const struct nlop_s* nlop_mri_gradient_step_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);

extern const struct nlop_s* mri_normal_inversion_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* mri_reg_proj_ker_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* mri_reg_pinv(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);
