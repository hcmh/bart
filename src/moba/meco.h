
struct linop_s;

struct nlop_s;

// struct noir_model_conf_s;

enum MODEL {
	WF,
	WFR2S,
	WF2R2S,
	R2S,
	PI,
};

extern long set_num_of_coeff(unsigned int sel_model);

extern void meco_calc_fat_modu(unsigned int N, const long dims[N], const complex float* TE, complex float* dst);

extern const struct linop_s* meco_get_fB0_trafo(struct nlop_s* op);
extern void meco_forw_fB0(const struct linop_s* op, complex float* dst, const complex float* src);
extern void meco_back_fB0(const struct linop_s* op, complex float* dst, const complex float* src);


extern struct nlop_s* nlop_meco_create(const int N, const long y_dims[N], const long x_dims[N], const complex float* TE, unsigned int sel_model, bool use_gpu);

