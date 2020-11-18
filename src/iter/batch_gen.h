#ifndef BATCHGEN_H
#define BATCHGEN_H

#include "misc/cppwrap.h"

enum BATCH_GEN_TYPE {BATCH_GEN_SAME, BATCH_GEN_SHUFFLE_BATCHES, BATCH_GEN_SHUFFLE_DATA, BATCH_GEN_RANDOM_DATA};
struct iter6_conf_s;
extern const struct nlop_s* batch_gen_create_from_iter(struct iter6_conf_s* iter_conf,long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc);

extern const struct nlop_s* batch_gen_create(long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc, enum BATCH_GEN_TYPE type, unsigned int seed);

extern const struct nlop_s* batch_gen_linear_create(long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc);
extern const struct nlop_s* batch_gen_rand_create(long D, long N, const long* dims[D], const _Complex float* data[__VLA(D)], long Nt, long Nc);

#endif