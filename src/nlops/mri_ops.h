
extern void reset_fft_op(void);

extern const struct nlop_s* nlop_mri_forward_create(const long dims[5], _Bool share_mask);
extern const struct nlop_s* nlop_mri_adjoint_create(const long dims[5], _Bool share_mask);

extern const struct nlop_s* nlop_gradient_step_unscaled_create(const long dims[5], _Bool share_mask);
extern const struct nlop_s* nlop_gradient_step_scaled_create(const long dims[5], _Bool share_mask);
extern const struct nlop_s* nlop_gradient_step_scaled_modular_create(const long dims[5], _Bool share_mask);