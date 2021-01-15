

struct modBlochFit;
extern void auto_scale(const struct modBlochFit* fit_para, float scale[4], const long ksp_dims[DIMS], complex float* kspace_data);

struct nlop_s;
void op_scaling(struct nlop_s* op, const long dims[DIMS], complex float* maps);
