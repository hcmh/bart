
struct nlop_s;
struct noir_model_conf_s;
extern struct nlop_s* nlop_T1MOLLI_create(int N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], 
                const complex float* TI1, const complex float* TI2, const complex float* TI_t1relax, bool use_gpu);

