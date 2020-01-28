

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
	int constrained_maps;
	int not_wav_maps;
};

void mdb_irgnm_l1(const struct mdb_irgnm_l1_conf* conf,
		const long dims[DIMS],
		struct nlop_s* nlop,
		long N, float* dst,
		long M, const float* src);

