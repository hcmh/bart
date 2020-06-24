
#include "nn/init.h"

enum MNIST_NETWORK_TYPE {MNIST_NETWORK_DEFAULT, MNIST_NETWORK_BATCHNORM};

extern int nn_mnist_get_num_weights(enum MNIST_NETWORK_TYPE type);
extern const struct nlop_s* get_nn_mnist(enum MNIST_NETWORK_TYPE type, int N_batch);
extern void init_nn_mnist(enum MNIST_NETWORK_TYPE type, _Complex float* weights);
extern void train_nn_mnist(enum MNIST_NETWORK_TYPE type, int N_batch, int N_total, _Complex float* weights, const _Complex float* in, const _Complex float* out, long epochs, _Bool adam);
extern void predict_nn_mnist(enum MNIST_NETWORK_TYPE type, int N_total, int N_batch, long prediction[__VLA(N_batch)], const _Complex float* weights, const _Complex float* in);
extern float accuracy_nn_mnist(enum MNIST_NETWORK_TYPE type, int N_total, int N_batch, const _Complex float* weights, const _Complex float* in, const _Complex float* out);
