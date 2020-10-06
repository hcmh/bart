

struct iter3_irgnm_conf;
struct nlop_s;


#ifndef DIMS
#define DIMS 16
#endif

struct mdb_irgnm_l1_conf {

	struct iter3_irgnm_conf* c2;

	int opt_reg;
	float step;
	float lower_bound;
	int constrained_maps; /*As bitmask*/
	bool auto_norm_off;
	int not_wav_maps;
	float wav_reg;
	unsigned int flags;
	bool usegpu;

	unsigned int algo;
	float rho;
	struct opt_reg_s* ropts;
};

const struct operator_p_s* T1inv_p_create(const struct mdb_irgnm_l1_conf* conf, const long dims[DIMS], struct nlop_s* nlop);

void mdb_irgnm_l1(const struct mdb_irgnm_l1_conf* conf,
		const long dims[DIMS],
		struct nlop_s* nlop,
		long N, float* dst,
		long M, const float* src);

