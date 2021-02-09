#include "nn/init.h"

struct segm_s {

	long Nb; // batchsize
	long imgx;
	long imgy;
	long classes;

	_Complex float* in_val;
	_Complex float* out_val;

	const char* val_loss;

};

const struct segm_s segm_default;

extern int nn_segm_get_num_weights(struct segm_s* segm);
extern nn_weights_t init_nn_segm_new(struct segm_s* segm);
extern const struct nlop_s* get_nn_segm(int N_batch, struct segm_s* segm);
extern void init_nn_segm(_Complex float* weights, struct segm_s* segm);
extern void train_nn_segm(	int N_total, int N_batch, int N_total_val, _Complex float* weights,
				const _Complex float* in, const _Complex float* out, long epochs, struct segm_s* segm);
extern void train_nn_segm_new(int N_total, int N_batch, nn_weights_t weights,
				const _Complex float* in, const _Complex float* out, long epochs, struct segm_s* segm);

extern void predict_nn_segm_new(int N_total, int N_batch, long* prediction, nn_weights_t weights, const _Complex float* in, struct segm_s* segm);
extern void predict_nn_segm(	int N_total, int N_batch, long* prediction,
				const _Complex float* weights, const _Complex float* in, struct segm_s* segm);

extern float accuracy_nn_segm_new(int N_total, int N_batch, nn_weights_t weights,
				const _Complex float* in, const _Complex float* out, struct segm_s* segm);
extern float accuracy_nn_segm(	int N_total, int N_batch, const _Complex float* weights,
				const _Complex float* in, const _Complex float* out, struct segm_s* segm);
