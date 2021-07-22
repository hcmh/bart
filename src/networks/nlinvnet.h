

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
	struct noir2_s* model;
	struct iter_conjgrad_conf* iter_conf;
	int iter_init;		//iterations with half update dx -> 0.5dx
	int iter_net;		//# of iterations with network
	int iter_net_shift;	//shift of iterations with network to earlier iterations

	_Bool low_mem;
	_Bool gpu;

	const char* graph_file;
};

extern struct nlinvnet_s nlinvnet_config_opts;

struct network_data_s;

extern void nlinvnet_init_varnet_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_varnet_test_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_resnet_default(struct nlinvnet_s* nlinvnet);

void nlinvnet_init_model_cart(struct nlinvnet_s* nlinvnet, int N,
	const long pat_dims[N], const _Complex float* pattern,
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N]);


enum nlinvnet_out { NLINVNET_OUT_CIM, NLINVNET_OUT_IMG, NLINVNET_OUT_IMG_COL };


void train_nlinvnet(struct nlinvnet_s* nlinvnet, int N, int Nb,
	const long cim_dims_trn[N], const _Complex float* ref_trn, const long ksp_dims_trn[N], const _Complex float* ksp_trn,
	const long cim_dims_val[N], const _Complex float* ref_val, const long ksp_dims_val[N], const _Complex float* ksp_val);

void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long img_dims[N], _Complex float* img,
	const long col_dims[N], _Complex float* col,
	const long ksp_dims[N], const _Complex float* ksp);

void eval_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const long cim_dims[N], const _Complex float* ref,
	const long ksp_dims[N], const _Complex float* ksp);
