
#include "misc/mri.h"

struct linop_s;
struct nlop_s;
struct noir_model_conf_s;

struct meco_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
	const struct linop_s* linop_fB0;
	const complex float* scaling;
};

extern struct meco_s meco_create(const long dims[DIMS], const long y_dims[DIMS], const long x_dims[DIMS], const complex float* mask, const complex float* TE, const complex float* psf, unsigned int sel_model, bool real_pd, unsigned int wgh_fB0, float scale_fB0, bool use_gpu, const struct noir_model_conf_s* conf);
