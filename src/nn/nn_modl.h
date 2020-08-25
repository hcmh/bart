#include "iter/iter6.h"
#include "iter/iter.h"

struct modl_s {

	long Nt; // number of steps of unrolled algorithm
	long Nl; // number of convolutions per DW
	long Nf; // number of filters

	long Kx; // filter size
	long Ky; // filter size
	long Kz; // filter size

	iter_conf* normal_inversion_iter_conf;
	_Bool batch_independent;
	float convergence_warn_limit;

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
	_Bool share_pattern;

	_Bool nullspace;
};

const struct modl_s modl_default;

extern void init_nn_modl(struct modl_s* modl);
extern void apply_nn_modl(	struct modl_s* modl,
				const long udims[5], _Complex float* out,
				const long kdims[5], const _Complex float* kspace, const _Complex float* coil, const long pdims[5], const _Complex float* pattern,
				_Bool normalize);
extern void apply_nn_modl_batchwise(	struct modl_s* modl,
					const long udims[5], _Complex float * out,
					const long kdims[5], const _Complex float* kspace, const _Complex float* coil, const long pdims[5], const _Complex float* pattern,
					long Nb, _Bool normalize);
extern void train_nn_modl(	struct modl_s* modl, struct iter6_conf_s* train_conf,
				const long udims[5], _Complex float* ref,
				const long kdims[5], _Complex float* kspace, const _Complex float* coil,
				const long pdims[5], const _Complex float* pattern,
				long Nb, _Bool random_order, _Bool normalize, const char** valid_files);

extern void nn_modl_store_weights(struct modl_s* modl, const char* name);
extern void nn_modl_load_weights(struct modl_s* modl, const char* name, _Bool overwrite_parameters);
extern void nn_modl_free_weights(struct modl_s* modl);

extern void nn_modl_move_gpucpu(struct modl_s* modl, _Bool gpu);