
struct nlop_s;
struct noir_model_conf_s;
struct modBlochFit;

extern struct nlop_s* nlop_Bloch_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long input_dims[N], const struct modBlochFit* fitPara, bool use_gpu);

