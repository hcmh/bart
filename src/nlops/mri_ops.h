extern const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], _Bool share_pattern);
extern const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], _Bool share_pattern);
extern const struct nlop_s* nlop_mri_normal_create(int N, const long dims[N], _Bool share_pattern);

extern const struct nlop_s* nlop_mri_gradient_step_create(int N, const long dims[N], _Bool share_pattern);

struct iter_conf_s;
extern const struct nlop_s* mri_normal_inversion_create(int N, const long dims[N], _Bool share_pattern, float lambda, _Bool batch_independent, float convergence_warn_limit, struct iter_conf_s* conf);
extern const struct nlop_s* mri_normal_inversion_create_with_lambda(int N, const long dims[N], _Bool share_pattern, float lambda, _Bool batch_independent, float convergence_warn_limit, struct iter_conf_s* conf);

extern const struct nlop_s* mri_reg_proj_ker_create(int N, const long dims[N], _Bool share_pattern, float lambda, _Bool batch_independent, float convergence_warn_limit, struct iter_conf_s* conf);
extern const struct nlop_s* mri_reg_proj_ker_create_with_lambda(int N, const long dims[N], _Bool share_pattern, float lambda, _Bool batch_independent, float convergence_warn_limit, struct iter_conf_s* conf);

extern const struct nlop_s* mri_reg_pinv(int N, const long dims[N], _Bool share_pattern, float lambda, _Bool batch_independent, float convergence_warn_limit, struct iter_conf_s* conf, _Bool rescale);
extern const struct nlop_s* mri_reg_pinv_with_lambda(int N, const long dims[N], _Bool share_pattern, float lambda, _Bool batch_independent, float convergence_warn_limit, struct iter_conf_s* conf, _Bool rescale);
