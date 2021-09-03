
struct config_nlop_mri_s;
struct iter_conjgrad_conf;

struct sense_normal_ops_s {

	int N;
	int ND;

	const long* col_dims;
	const long* psf_dims;

	const long* img_dims;
	const long* bat_dims;
	const long* img_dims_slice;

	unsigned long bat_flag;
	unsigned long img_flag;

	_Bool noncart;

	struct sense_noncart_normal_s* sense_noncart;
	struct sense_cart_normal_s* sense_cart;
};

struct sense_normal_ops_s* sense_normal_create(int N, const long max_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf);

void sense_normal_free(struct sense_normal_ops_s* d);
void sense_normal_update_ops(struct sense_normal_ops_s* d, int N, const long col_dims[N], const _Complex float* coils, int ND, const long psf_dims[ND], const _Complex float* psf);

const struct operator_s* sense_get_normal_op(struct sense_normal_ops_s* d, int N, const long pos[N]);
void sense_apply_normal_ops(struct sense_normal_ops_s* d, int N, const long img_dims[N], _Complex float* dst, const _Complex float* src);
void sense_apply_normal_inv(struct sense_normal_ops_s* d, const struct iter_conjgrad_conf* iter_conf, int N, const long img_dims[N], _Complex float* dst, const _Complex float* src);


