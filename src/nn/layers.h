#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "nlops/conv.h"

enum NETWORK_STATUS {STAT_TRAIN, STAT_TEST};
extern enum NETWORK_STATUS network_status;

extern const struct nlop_s* append_dense_layer(const struct nlop_s* network, int o, int out_neurons);

extern const struct nlop_s* append_conv_layer(const struct nlop_s* network, int o, int filters, const long kernel_size[3], enum CONV_PAD conv_pad, _Bool channel_first, const long strides[3], const long dilations[3]);
extern const struct nlop_s* append_maxpool_layer(const struct nlop_s* network, int o, const long pool_size[3], enum CONV_PAD conv_pad, _Bool channel_first);

extern const struct nlop_s* append_dropout_layer(const struct nlop_s* network, int o, float p);
extern const struct nlop_s* append_flatten_layer(const struct nlop_s* network, int o);

#endif