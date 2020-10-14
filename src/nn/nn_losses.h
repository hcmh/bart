#ifndef NN_LOSSES_H
#define NN_LOSSES_H

#include "nn/nn.h"

extern nn_t nn_loss_mse_append(nn_t network, int o, const char* oname, unsigned long mean_dims);
extern nn_t nn_loss_cce_append(nn_t network, int o, const char* oname);

#endif