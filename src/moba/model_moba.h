


#include "misc/mri.h"

struct linop_s;
struct nlop_s;
struct noir_model_conf_s;
struct moba_conf_s;

struct moba_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
	const struct linop_s* linop_alpha;
};


extern struct moba_s moba_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf, const struct moba_conf_s* conf_model, _Bool usegpu);


