
#include "nn/init.h"

extern int nn_mnist_get_num_weights(void);
extern const struct nlop_s* get_nn_mnist(int N_batch);
extern void init_nn_mnist(_Complex float* weights);
extern void train_nn_mnist(int N_batch, int N_total, _Complex float* weights, const _Complex float* in, const _Complex float* out, long epochs);
extern void predict_nn_mnist(int N_batch, long prediction[__VLA(N_batch)], const _Complex float* weights, const _Complex float* in);
extern float accuracy_nn_mnist(int N_batch, const _Complex float* weights, const _Complex float* in, const _Complex float* out);





