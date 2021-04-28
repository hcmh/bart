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

	_Bool shared_weights;

	struct nn_weights_s* weights;

	float lambda_init;
	float init_scale_mu;

	_Bool normalize;

	_Bool share_pattern;

	_Bool init_tickhonov;
	float lambda_fixed_tickhonov;

	_Bool low_mem;

	_Bool regrid;

	_Bool monitor_lambda;
};

extern const struct vn_s vn_default;

extern void apply_vn(	struct vn_s* vn,
 			const long idims[5], _Complex float* out,
			const long kdims[5], const _Complex float* kspace,
			const long cdims[5], const _Complex float* coil,
			const long pdims[5], const _Complex float* pattern);

extern void apply_vn_batchwise(	struct vn_s* vn,
 				const long idims[5], _Complex float* out,
				const long kdims[5], const _Complex float* kspace,
				const long cdims[5], const _Complex float* coil,
				const long pdims[5], const _Complex float* pattern,
				long Nb);

struct iter6_conf_s;
struct network_data_s;

extern void train_vn(	struct vn_s* vn, struct iter6_conf_s* train_conf,
			const long idims[5], _Complex float* ref,
			const long kdims[5], _Complex float* kspace,
			const long cdims[5], const _Complex float* coil,
			const long pdims[5], const _Complex float* pattern,
			long Nb, struct network_data_s* valid_files);

extern void init_vn(struct vn_s* vn);
extern void load_vn(struct vn_s* vn, const char* filename, _Bool overwrite_pars);