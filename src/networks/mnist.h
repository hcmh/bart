
#include "nn/init.h"
#include "nn/layers.h"
#include "nn/nn.h"
#include "nn/weights.h"

extern nn_weights_t init_nn_mnist(void);
extern void train_nn_mnist(int N_batch, int N_total, nn_weights_t, const _Complex float* in, const _Complex float* out, long epochs);
extern void predict_nn_mnist(int N_total, int N_batch, long prediction[N_total], nn_weights_t weights, const _Complex float* in);
extern float accuracy_nn_mnist(int N_total, int N_batch, nn_weights_t weights, const _Complex float* in, const _Complex float* out);
