extern const struct nlop_s* nlop_stats_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern const struct nlop_s* nlop_normalize_create(int N, const long dims[__VLA(N)], unsigned long flags, float epsilon);
extern const struct nlop_s* nlop_scale_and_shift_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern const struct nlop_s* nlop_batchnorm_floatingstats_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern const struct nlop_s* nlop_batchnorm_create(int N, const long dims[N], unsigned long flags, float epsilon);
