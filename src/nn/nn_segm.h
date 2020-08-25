#ifndef IMG_DIM
#define IMG_DIM 256	//dimension of square input images
#endif

#ifndef MASK_DIM
#define MASK_DIM 4	//number of dimensions of segmentation masks
#endif

#include "nn/init.h"

extern int nn_segm_get_num_weights(void);
extern const struct nlop_s* get_nn_segm(int N_batch);
extern void init_nn_segm(_Complex float* weights);
extern void train_nn_segm(int N_batch, int N_total, int N_total_val, _Complex float* weights, const _Complex float* in, const _Complex float* out, const _Complex float* in_val, const _Complex float* out_val,long epochs);
extern void predict_nn_segm(int N_total, int N_batch, long prediction[(IMG_DIM * IMG_DIM * N_batch)], const _Complex float* weights, const _Complex float* in);
extern float accuracy_nn_segm(int N_total, int N_batch, const _Complex float* weights, const _Complex float* in, const _Complex float* out);
