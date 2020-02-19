#ifndef NN_OPS_H
#define NN_OPS_H

extern const struct nlop_s* nlop_maxpool_create(int N, const long dims[__VLA(N)], const long pool_size[__VLA(N)]);
extern const struct nlop_s* nlop_dropout_create(int N, const long dims[__VLA(N)], float p, unsigned int shared_dims_flag);

#endif
