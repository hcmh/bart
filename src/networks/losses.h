
struct loss_config_s {

	float weighting_mse_sa;
	float weighting_mse;
	float weighting_psnr;
	float weighting_ssim;

	float weighting_cce;
	float weighting_accuracy;

	float weighting_dice0;
	float weighting_dice1;
	float weighting_dice2;

	unsigned int label_index;
};

extern struct loss_config_s loss_empty;

extern struct loss_config_s loss_mse;
extern struct loss_config_s loss_mse_sa;
extern struct loss_config_s loss_image_valid;

extern struct loss_config_s loss_classification;
extern struct loss_config_s loss_classification_valid;

extern const struct nn_s* loss_create(const struct loss_config_s* config, unsigned int N, const long dims[N], _Bool combine);
