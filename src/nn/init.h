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
