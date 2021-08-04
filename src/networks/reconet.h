

struct nn_weights_s;
struct loss_config_s;

enum BOOL_SELECT {BOOL_DEFAULT, BOOL_TRUE, BOOL_FALSE};

struct reconet_s {

	struct network_s* network;
	_Bool kspace;

	long Nt;

	enum BOOL_SELECT share_weights_select;
	enum BOOL_SELECT share_lambda_select;
	_Bool share_weights;
	_Bool share_lambda;

	_Bool reinsert;

	struct config_nlop_mri_s* mri_config;

	//data consistency config
	float dc_lambda_fixed;
	float dc_lambda_init;
	_Bool dc_gradient;
	_Bool dc_scale_max_eigen;
	_Bool dc_tickhonov;
	int dc_max_iter;

	//network initialization
	_Bool normalize;
	_Bool tickhonov_init;
	int init_max_iter;
	float init_lambda_fixed;
	float init_lambda_init;

	struct nn_weights_s* weights;
	struct iter6_conf_s* train_conf;

	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;
	float multi_loss;

	_Bool low_mem;
	_Bool gpu;

	const char* graph_file;

	_Bool rss_scale;
	_Bool normalize_rss;
};

extern struct reconet_s reconet_config_opts;

struct network_data_s;

extern void reconet_init_modl_default(struct reconet_s* reconet);
extern void reconet_init_varnet_default(struct reconet_s* reconet);
extern void reconet_init_unet_default(struct reconet_s* reconet);

extern void reconet_init_modl_test_default(struct reconet_s* reconet);
extern void reconet_init_varnet_test_default(struct reconet_s* reconet);
extern void reconet_init_unet_test_default(struct reconet_s* reconet);

extern void apply_reconet(	const struct reconet_s* reconet, unsigned int N, const long max_dims[N],
				const long img_dims[N], _Complex float* out, const _Complex float* adjoint,
				const long col_dims[N], const _Complex float* coil,
				int ND, const long psf_dims[ND], const _Complex float* psf);

extern void train_reconet(	struct reconet_s* reconet, unsigned int N, const long max_dims[N],
				const long img_dims[N], _Complex float* ref, const _Complex float* adjoint,
				const long col_dims[N], const _Complex float* coil,
				int ND, const long psf_dims[ND], const _Complex float* psf,
				long Nb, struct network_data_s* valid_files);

extern void eval_reconet(	const struct reconet_s* reconet, unsigned int N, const long max_dims[N],
				const long img_dims[N], const _Complex float* ref, const _Complex float* adjoint,
				const long col_dims[N], const _Complex float* coil,
				int ND, const long psf_dims[N], const _Complex float* psf);