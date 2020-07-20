
extern const struct operator_p_s* operator_project_pos_real_create(long N, const long dims[N]);
extern const struct operator_p_s* operator_project_mean_free_create(long N, const long dims[N], unsigned long bflag);
extern const struct operator_p_s* operator_project_sphere_create(long N, const long dims[N], unsigned long bflag, _Bool real);
extern const struct operator_p_s* operator_project_mean_free_sphere_create(long N, const long dims[N], unsigned long bflag, _Bool real);
extern const struct operator_p_s* operator_project_real_interval_create(long N, const long dims[N], float min, float max);