
struct nlop_s;
struct noir_model_conf_s;
struct modBlochFit;

extern struct nlop_s* nlop_Bloch_create(int N, const long der_dims[N], const long map_dims[N], const long out_dims[N], const long in_dims[N], const struct modBlochFit* fit_para, bool use_gpu);

extern complex float* blochfun_get_derivatives(struct nlop_s* op);
