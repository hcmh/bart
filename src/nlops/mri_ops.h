extern const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], _Bool share_pattern);
extern const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], _Bool share_pattern);
extern const struct nlop_s* nlop_mri_normal_create(int N, const long dims[N], _Bool share_pattern);

extern const struct nlop_s* nlop_mri_gradient_step_create(int N, const long dims[N], _Bool share_pattern);