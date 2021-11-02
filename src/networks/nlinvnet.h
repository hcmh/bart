

struct nn_weights_s;
struct loss_config_s;

struct nlinvnet_s {

	struct network_s* network;

	_Bool share_weights;

	struct nn_weights_s* weights;
	struct iter6_conf_s* train_conf;

	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;

	struct noir2_conf_s* conf;

	int Nb;
	struct noir2_s** models;
	struct noir2_s* model_valid;

	struct iter_conjgrad_conf* iter_conf;
	struct iter_conjgrad_conf* iter_conf_net;
	float cgtol;
	int iter_init;		//iterations with half update dx -> 0.5dx
	int iter_net;		//# of iterations with network

	_Bool low_mem;
	_Bool gpu;

	_Bool normalize_rss;
	_Bool fix_lambda;

	_Bool ksp_training;
	float ksp_split;
	float ksp_noise;
	unsigned long ksp_shared_dims;
	_Bool ksp_ref_net_only;

	float scaling;

	float l1_norm;
	float l2_norm;

	_Bool ref;

	const char* graph_file;
};

extern struct nlinvnet_s nlinvnet_config_opts;

struct network_data_s;

extern void nlinvnet_init_varnet_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_varnet_test_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_resnet_default(struct nlinvnet_s* nlinvnet);

void nlinvnet_init_model_cart(struct nlinvnet_s* nlinvnet, int N,
	const long pat_dims[N],
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N]);

void nlinvnet_init_model_noncart(struct nlinvnet_s* nlinvnet, int N,
	const long trj_dims[N],
	const long wgh_dims[N],
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N]);


enum nlinvnet_out { NLINVNET_OUT_CIM, NLINVNET_OUT_KSP, NLINVNET_OUT_IMG_COL, NLINVNET_OUT_REF };

struct named_data_list_s;
void train_nlinvnet(struct nlinvnet_s* nlinvnet, int Nb, struct named_data_list_s* train_data, struct named_data_list_s* valid_data);

void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long img_dims[N], _Complex float* img,
	const long col_dims[N], _Complex float* col,
	const long ksp_dims[N], const _Complex float* ksp,
	const long pat_dims[N], const _Complex float* pat,
	const long trj_dims[N], const _Complex float* trj,
	const long ref_img_dims[N], const _Complex float* ref_img,
	const long ref_col_dims[N], const _Complex float* ref_col);

void eval_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long cim_dims[N], const _Complex float* ref,
	const long ksp_dims[N], const _Complex float* ksp,
	const long pat_dims[N], const _Complex float* pat,
	const long trj_dims[N], const _Complex float* trj,
	const long ref_img_dims[N], const _Complex float* ref_img,
	const long ref_col_dims[N], const _Complex float* ref_col);
