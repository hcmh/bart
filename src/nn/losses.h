
extern const struct nlop_s* nlop_mse_create(int N, const long dims[__VLA(N)], unsigned long mean_dims);
extern const struct nlop_s* nlop_mpsnr_create(int N, const long dims[__VLA(N)], unsigned long mean_dims);
extern const struct nlop_s* nlop_mssim_create(int N, const long dims[__VLA(N)], const long kdims[__VLA(N)], unsigned long conv_dims);
extern const struct nlop_s* nlop_cce_create(int N, const long dims[__VLA(N)], unsigned long scaling_flag);
extern const struct nlop_s* nlop_weighted_cce_create(int N, const long dims[__VLA(N)], unsigned long batch_flag);
