#include "iter/iter6.h"

struct unet_s;

struct nullspace_s {

	long Nb; // batchsize

	_Complex float* lambda;
	float lambda_init;
	float lambda_min;
	float lambda_max;
	float lambda_fixed;

	_Bool share_mask;

	struct unet_s* unet;

	_Bool rescale;
	_Bool nullspace;
};

const struct nullspace_s nullspace_default;

extern void compute_zero_filled(	const long udims[5], _Complex float * out,
					const long kdims[5], const _Complex float* kspace, const _Complex float* coil, long mdims[5], const _Complex float* mask);
extern void compute_reference(	const long udims[5], _Complex float * ref,
				const long kdims[5], const _Complex float* kspace, const _Complex float* coil);

extern void init_nn_nullspace(struct nullspace_s* nullspace, long udims[5]);
extern void apply_nn_nullspace(	struct nullspace_s* nullspace,
				const long idims[5], _Complex float* out,
				const long kdims[5], const _Complex float* kspace,
				const long cdims[5], const _Complex float* coil,
				const long mdims[5], const _Complex float* mask);
extern void train_nn_nullspace(	struct nullspace_s* nullspace, iter6_conf* train_conf,
				const long idims[5], const _Complex float* ref,
				const long kdims[5], const _Complex float* kspace,
				const long cdims[5], const _Complex float* coil,
				const long mdims[5], const _Complex float* mask,
				_Bool random_order, const char* history_filename, const char** valid_files);

extern void nn_nullspace_store_weights(struct nullspace_s* nullspace, const char* name);
extern void nn_nullspace_load_weights(struct nullspace_s* nullspace, const char* name);
extern void nn_nullspace_free_weights(struct nullspace_s* nullspace);

extern void nn_nullspace_move_gpucpu(struct nullspace_s* nullspace, _Bool gpu);
