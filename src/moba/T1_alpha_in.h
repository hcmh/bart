
struct linop_s;
struct nlop_s;
struct noir_model_conf_s;

extern struct nlop_s* nlop_T1_alpha_in_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N],
                const long TI_dims[N], const complex float* TI, const complex float* alpha, bool use_gpu);

