struct conf_mri_dims {

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
	struct iter_conjgrad_conf* iter_conf;
};

struct conf_mri_dims conf_nlop_mri_simple;

extern const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], const struct conf_mri_dims* conf);
extern const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], const struct conf_mri_dims* conf);
extern const struct nlop_s* nlop_mri_normal_create(int N, const long dims[N], const struct conf_mri_dims* conf);

extern const struct nlop_s* nlop_mri_gradient_step_create(int N, const long dims[N], const struct conf_mri_dims* conf);

extern const struct nlop_s* mri_normal_inversion_create(int N, const long dims[N], const struct conf_mri_dims* conf);
extern const struct nlop_s* mri_reg_proj_ker_create(int N, const long dims[N], const struct conf_mri_dims* conf);
extern const struct nlop_s* mri_reg_pinv(int N, const long dims[N], const struct conf_mri_dims* conf);
