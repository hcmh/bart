
struct nlop_s;
struct noir_model_conf_s;
struct modBlochFit;

extern const struct linop_s* Bloch_get_alpha_trafo(struct nlop_s* op);
extern void Bloch_forw_alpha(const struct linop_s* op, complex float* dst, const complex float* src);
extern void Bloch_back_alpha(const struct linop_s* op, complex float* dst, const complex float* src);

extern struct nlop_s* nlop_Bloch_create2(int N, const long der_dims[N], const long map_dims[N], const long out_dims[N], const long in_dims[N], const struct modBlochFit* fit_para, bool use_gpu);

