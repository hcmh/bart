

struct iter3_conf_s;
struct iter_op_s;

typedef struct iter3_conf_s iter3_conf;


struct operator_p_s;
struct iter_nlop_s;
struct nlop_s;

void iter4_irgnm_l1(const struct iter3_conf_s* _conf,
		const long dims[],
		struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* solve,
		const struct iter_op_s cb);

