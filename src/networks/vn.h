struct nn_weights_s;
struct vn_s {

	long Nl;
	long Nf;
	long Kx;
	long Ky;
	long Kz;

	long Nw;

	float Imax;
	float Imin;

	struct nn_weights_s* weights;

	float lambda_init;
	float init_scale_mu;

	_Bool normalize;

	_Bool share_pattern;

	_Bool init_tickhonov;
	float lambda_fixed_tickhonov;
	float lambda_init_tickhonov;
};

extern const struct vn_s vn_default;

extern void apply_vn(	struct vn_s* vn,
 			const long udims[5], _Complex float* out,
			const long kdims[5], const _Complex float* kspace, const _Complex float* coil,
			const long pdims[5], const _Complex float* pattern);
extern void apply_vn_batchwise(	struct vn_s* vn,
 				const long udims[5], _Complex float* out,
				const long kdims[5], const _Complex float* kspace, const _Complex float* coil,
				const long pdims[5], const _Complex float* pattern,
				long Nb);

struct iter6_conf_s;

extern void train_vn(	struct vn_s* vn, struct iter6_conf_s* train_conf,
			const long udims[5], _Complex float* ref,
			const long kdims[5], _Complex float* kspace, const _Complex float* coil,
			const long pdims[5], const _Complex float* pattern,
			long Nb, const char** valid_files);

extern void init_vn(struct vn_s* vn);
extern void load_vn(struct vn_s* vn, const char* filename, _Bool overwrite_pars);
