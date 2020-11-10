#ifndef NN_OPS_H
#define NN_OPS_H

#include "nn/layers.h"

extern const struct nlop_s* nlop_maxpool_create(int N, const long dims[__VLA(N)], const long pool_size[__VLA(N)]);
extern const struct nlop_s* nlop_dropout_create(int N, const long dims[__VLA(N)], float p, unsigned int shared_dims_flag);
extern const struct linop_s* linop_avgpool_create(int N, const long dims[__VLA(N)], const long pool_size[__VLA(N)]);
extern const struct nlop_s* nlop_zmax_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern const struct linop_s* linop_pool_create(int N, const long dims[__VLA(N)], const long pool_size[__VLA(N)]);
extern const struct nlop_s* nlop_blurpool_create(int N, const long dims[__VLA(N)], const long pool_size[__VLA(N)]);
extern const struct nlop_s* nlop_norm_zmax_create(int N, const long dims[__VLA(N)], unsigned long batch_flag, _Bool abs);

#endif
