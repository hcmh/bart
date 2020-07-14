#include "iter/iter6.h"

struct vn_s {

	long Nl;

	long Nf;

	long Kx;
	long Ky;
	long Kz;

	long Px;
	long Py;
	long Pz;

	long Nw;
	float Imax;
	float Imin;

	_Complex float* conv_w;
	_Complex float* rbf_w;
	_Complex float* lambda_w;

	_Complex float* conv_w_cpu;
	_Complex float* rbf_w_cpu;
	_Complex float* lambda_w_cpu;

	float lambda_init;
	float init_scale_mu;

	_Bool share_mask;
};

const struct vn_s vn_default;

extern void compute_zero_filled(	const long udims[5], _Complex float * out,
					const long kdims[5], const _Complex float* kspace, const _Complex float* coil, long mdims[5], const _Complex float* mask);
extern void compute_reference(	const long udims[5], _Complex float * ref,
				const long kdims[5], const _Complex float* kspace, const _Complex float* coil);

extern void apply_variational_network(	struct vn_s* vn,
 					const long udims[5], _Complex float* out,
					const long kdims[5], const _Complex float* kspace, const _Complex float* coil, const long mdims[5], const _Complex float* mask);
extern void apply_variational_network_batchwise(	struct vn_s* vn,
 							const long udims[5], _Complex float* out,
							const long kdims[5], const _Complex float* kspace, const _Complex float* coil, const long mdims[5], const _Complex float* mask,
							long Nb);

extern void train_nn_varnet(	struct vn_s* vn, iter6_conf* train_conf,
				const long udims[5], const _Complex float* ref,
				const long kdims[5], const _Complex float* kspace, const _Complex float* coil,
				const long mdims[5], const _Complex float* mask,
				long Nb, _Bool random_order, const char* history_filename, const char** valid_files);

extern void initialize_varnet(struct vn_s* vn);

extern void compute_scale(const long dims[5], _Complex float* scaling, const _Complex float * u0);
extern void normalize_max(const long dims[5], const _Complex float* scaling, _Complex float* out, const _Complex float* in);
extern void renormalize_max(const long dims[5], const _Complex float* scaling, _Complex float* out, const _Complex float* in);

extern void unmask_zeros(long mdims[5], _Complex float* mask, long kdims[5], const _Complex float* kspace);

extern void vn_move_gpucpu(struct vn_s* vn, _Bool gpu);
extern void save_varnet(struct vn_s* vn, const char* filename_conv_w, const char* filename_rbf_w, const char* filename_lambda);