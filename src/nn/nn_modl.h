#include "iter/iter6.h"

struct modl_s {

	long Nb; // batchsize

	long Nt; // number of steps of unrolled algorithm
	long Nl; // number of convolutions per DW
	long Nf; // number of filters


	long Kx; // filter size
	long Ky; // filter size
	long Kz; // filter size

	_Complex float* lambda;

	_Complex float* conv_0;
	_Complex float* conv_i;
	_Complex float* conv_n;

	_Complex float* bias_0;
	_Complex float* bias_i;
	_Complex float* bias_n;

	_Complex float* gamma_n;

	_Complex float* bn_0;
	_Complex float* bn_i;
	_Complex float* bn_n;

	float lambda_init;
	float lambda_min;
	float lambda_max;

	float lambda_fixed;

	_Bool shared_weights;
	_Bool shared_lambda;
	_Bool share_mask;

	_Bool nullspace;
};

const struct modl_s modl_default;

extern void compute_zero_filled(	const long udims[5], _Complex float * out,
					const long kdims[5], const _Complex float* kspace, const _Complex float* coil, long mdims[5], const _Complex float* mask);
extern void compute_reference(	const long udims[5], _Complex float * ref,
				const long kdims[5], const _Complex float* kspace, const _Complex float* coil);

extern void init_nn_modl(struct modl_s* modl);
extern void apply_nn_modl(	struct modl_s* modl,
				const long idims[5], _Complex float* out,
				const long kdims[5], const _Complex float* kspace,
				const long cdims[5], const _Complex float* coil,
				const long mdims[5], const _Complex float* mask);
extern void train_nn_modl(	struct modl_s* modl, iter6_conf* train_conf,
				const long idims[5], const _Complex float* ref,
				const long kdims[5], const _Complex float* kspace,
				const long cdims[5], const _Complex float* coil,
				const long mdims[5], const _Complex float* mask,
				_Bool random_order, const char* history_filename, const char** valid_files);

extern void nn_modl_store_weights(struct modl_s* modl, const char* name);
extern void nn_modl_load_weights(struct modl_s* modl, const char* name);
extern void nn_modl_free_weights(struct modl_s* modl);

extern void nn_modl_move_gpucpu(struct modl_s* modl, _Bool gpu);

#if 0
extern void compute_scale(const long dims[5], _Complex float* scaling, const _Complex float * u0);
extern void normalize_max(const long dims[5], const _Complex float* scaling, _Complex float* out, const _Complex float* in);
extern void renormalize_max(const long dims[5], const _Complex float* scaling, _Complex float* out, const _Complex float* in);

extern void unmask_zeros(long mdims[5], _Complex float* mask, long kdims[5], const _Complex float* kspace);
#endif