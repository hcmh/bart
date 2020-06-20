#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "nlops/conv.h"

enum NETWORK_STATUS {STAT_TRAIN, STAT_TEST};
extern enum NETWORK_STATUS network_status;

extern const struct nlop_s* append_dense_layer(const struct nlop_s* network, int o, int out_neurons);

extern const struct nlop_s* append_convcorr_layer(const struct nlop_s* network, int o, int filters, const long kernel_size[3], _Bool conv, enum PADDING conv_pad, _Bool channel_first, const long strides[3], const long dilations[3]);
extern const struct nlop_s* append_transposed_convcorr_layer(const struct nlop_s* network, int o, int channels, long const kernel_size[3], _Bool conv, _Bool adjoint, enum PADDING conv_pad, _Bool channel_first, const long strides[3], const long dilations[3]);
extern const struct nlop_s* append_maxpool_layer(const struct nlop_s* network, int o, const long pool_size[3], enum PADDING conv_pad, _Bool channel_first);

extern const struct nlop_s* append_padding_layer(const struct nlop_s* network, int o, long N, long pad_for[__VLA(N)], long pad_after[__VLA(N)], enum PADDING pad_type);

extern const struct nlop_s* append_dropout_layer(const struct nlop_s* network, int o, float p);
extern const struct nlop_s* append_flatten_layer(const struct nlop_s* network, int o);

extern const struct nlop_s* append_batchnorm_layer(const struct nlop_s* network, int o, unsigned long norm_flags);

#endif