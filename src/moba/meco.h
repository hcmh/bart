
struct linop_s;

struct nlop_s;

// struct noir_model_conf_s;

enum MECO_MODEL {
	MECO_WF,
	MECO_WFR2S,
	MECO_WF2R2S,
	MECO_R2S,
	MECO_PHASEDIFF,
	MECO_PI,
};

enum MECO_WEIGHT_fB0 {
	MECO_IDENTITY,
	MECO_SOBOLEV,
};

extern long set_num_of_coeff(unsigned int sel_model);

extern long set_PD_flag(unsigned int sel_model);
extern long set_R2S_flag(unsigned int sel_model);
extern long set_fB0_flag(unsigned int sel_model);


extern void meco_calc_fat_modu(unsigned int N, const long dims[N], const complex float* TE, complex float* dst);

extern const complex float* meco_get_scaling(struct nlop_s* op);
extern const struct linop_s* meco_get_fB0_trafo(struct nlop_s* op);
extern void meco_forw_fB0(const struct linop_s* op, complex float* dst, const complex float* src);
extern void meco_back_fB0(const struct linop_s* op, complex float* dst, const complex float* src);


extern struct nlop_s* nlop_meco_create(const int N, const long y_dims[N], const long x_dims[N], const complex float* TE, unsigned int sel_model, bool real_pd, unsigned int wgh_fB0, float scale_fB0, _Bool use_gpu);

