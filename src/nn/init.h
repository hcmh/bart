#ifndef INITIALIZER_H
#define INITIALIZER_H

struct initializer_s;
typedef void (*initializer_f)(const struct initializer_s* conf, long N, const long dims[N], _Complex float* weights);

void initializer_apply(const struct initializer_s* conf, long N, const long dims[N], _Complex float* weights);
void initializer_free(const struct initializer_s* conf);
const struct initializer_s* initializer_clone(const struct initializer_s* x);

unsigned long in_flag_conv(_Bool c1);
unsigned long out_flag_conv(_Bool c1);

const struct initializer_s* init_const_create(_Complex float val);
const struct initializer_s* init_xavier_create(unsigned long in_flags, unsigned long out_flags, _Bool real, _Bool uniform);
const struct initializer_s* init_kaiming_create(unsigned long in_flags, _Bool real, _Bool uniform, float leaky_val);

const struct initializer_s* init_std_normal_create(_Bool real, float scale, float mean);
const struct initializer_s* init_uniform_create(_Bool real, float scale, float mean);

const struct initializer_s* init_linspace_create(unsigned int dim, _Complex float min_val, _Complex float max_val, _Bool max_inc);

typedef _Complex float* init_f(long N, const long* dims, _Complex float* src, _Bool c1);

_Complex float* init_kaiming_uniform_conv(long N, const long* kernel_dims, _Complex float* src, _Bool c1, float leaky_val);
_Complex float* init_kaiming_uniform_conv_complex(long N, const long* kernel_dims, _Complex float* src, _Bool c1, float leaky_val);
_Complex float* init_kaiming_normal_conv_complex(long N, const long* kernel_dims, _Complex float* src, _Bool c1, float leaky_val);
_Complex float* init_kaiming_normal_conv(long N, const long* kernel_dims, _Complex float* src, _Bool c1, float leaky_val);

init_f init_glorot_uniform_dense;
init_f init_glorot_uniform_conv;
init_f init_glorot_uniform_conv_complex;
init_f init_bias;
init_f init_auto;
#endif