

struct nn_weights_s;
struct loss_config_s;
struct reconet_s {

	struct network_s* network;

	long Nt;

	_Bool reinsert;
	_Bool share_weights;
	_Bool share_lambda;

	struct config_nlop_mri_s* mri_config;
	struct config_nlop_mri_dc_s* mri_config_dc;
	_Bool dc_tickhonov;
	_Bool dc_gradient;

	_Bool tickhonov_init;
	unsigned long normalize;
	struct config_nlop_mri_dc_s* mri_config_dc_init;

	struct nn_weights_s* weights;
	struct iter6_conf_s* train_conf;

	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;

	_Bool low_mem;
	_Bool gpu;
};

extern struct reconet_s reconet_init;

struct network_data_s;

extern void reconet_init_modl_default(struct reconet_s* reconet);
extern void reconet_init_varnet_default(struct reconet_s* reconet);

extern void reconet_init_modl_test_default(struct reconet_s* reconet);
extern void reconet_init_varnet_test_default(struct reconet_s* reconet);

extern void apply_reconet(	const struct reconet_s* reconet, unsigned int N,
				const long idims[N], _Complex float* out,
				const long kdims[N], const _Complex float* kspace,
				const long cdims[N], const _Complex float* coil,
				const long pdims[N], const _Complex float* pattern);

extern void apply_reconet_batchwise(	const struct reconet_s* reconet, unsigned int N,
					const long idims[N], _Complex float* out,
					const long kdims[N], const _Complex float* kspace,
					const long cdims[N], const _Complex float* coil,
					const long pdims[N], const _Complex float* pattern,
					long Nb);

extern void train_reconet(	struct reconet_s* reconet, unsigned int N,
				const long idims[N], _Complex float* ref,
				const long kdims[N], _Complex float* kspace,
				const long cdims[N], const _Complex float* coil,
				const long pdims[N], const _Complex float* pattern,
				long Nb, struct network_data_s* valid_files);