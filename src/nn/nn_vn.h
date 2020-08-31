struct vn_s {

	long Nl;
	long Nf;
	long Kx;
	long Ky;
	long Kz;

	long Nw;

	float Imax;
	float Imin;

	_Complex float* conv_w;
	_Complex float* rbf_w;
	_Complex float* lambda_w;

	float lambda_init;
	float init_scale_mu;

	_Bool share_pattern;
};

extern const struct vn_s vn_default;

extern void apply_variational_network(	struct vn_s* vn,
 					const long udims[5], _Complex float* out,
					const long kdims[5], const _Complex float* kspace, const _Complex float* coil, const long pdims[5], const _Complex float* pattern,
					_Bool normalize);
extern void apply_variational_network_batchwise(	struct vn_s* vn,
 							const long udims[5], _Complex float* out,
							const long kdims[5], const _Complex float* kspace, const _Complex float* coil, const long pdims[5], const _Complex float* pattern,
							long Nb, _Bool normalize);
struct iter6_conf_s;
extern void train_nn_varnet(	struct vn_s* vn, struct iter6_conf_s* train_conf,
				const long udims[5], _Complex float* ref,
				const long kdims[5], _Complex float* kspace, const _Complex float* coil,
				const long pdims[5], const _Complex float* pattern,
				long Nb, _Bool random_order, _Bool normalize, const char** valid_files);

extern void initialize_varnet(struct vn_s* vn);
extern void save_varnet(struct vn_s* vn, const char* filename);
extern void load_varnet(struct vn_s* vn, const char* filename);
extern void vn_move_gpucpu(struct vn_s* vn, _Bool gpu);
