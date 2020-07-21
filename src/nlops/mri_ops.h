
extern void reset_fft_op(void);

extern const struct nlop_s* nlop_mri_forward_create(const long dims[5], _Bool share_mask);
extern const struct nlop_s* nlop_mri_adjoint_create(const long dims[5], _Bool share_mask);

extern const struct nlop_s* nlop_gradient_step_unscaled_create(const long dims[5], _Bool share_mask);
extern const struct nlop_s* nlop_gradient_step_scaled_create(const long dims[5], _Bool share_mask);
extern const struct nlop_s* nlop_gradient_step_scaled_modular_create(const long dims[5], _Bool share_mask);

extern const struct nlop_s* mri_normal_inversion_create_general(int N, long dims[__VLA(N)],
								unsigned long iflags, unsigned long cflags, unsigned long mflags, unsigned long ciflags, unsigned long fftflags,
								float lambda);
extern const struct nlop_s* mri_normal_inversion_create_general_with_lambda(int N, long dims[__VLA(N)],
								unsigned long iflags, unsigned long cflags, unsigned long mflags, unsigned long ciflags, unsigned long fftflags,
								float lambda);

extern const struct nlop_s* mri_reg_projection_kerT_create_general(	int N, long dims[__VLA(N)],
									unsigned long iflags, unsigned long cflags, unsigned long mflags, unsigned long ciflags, unsigned long fftflags,
									float lambda);
extern const struct nlop_s* mri_reg_projection_ker_create_general(	int N, long dims[__VLA(N)],
									unsigned long iflags, unsigned long cflags, unsigned long mflags, unsigned long ciflags, unsigned long fftflags,
									float lambda);
extern const struct nlop_s* mri_reg_projection_ker_create_general_with_lambda(	int N, long dims[__VLA(N)],
									unsigned long iflags, unsigned long cflags, unsigned long mflags, unsigned long ciflags, unsigned long fftflags,
									float lambda);
