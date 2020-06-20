typedef _Complex float* init_f(long N, const long* dims, _Complex float* src, _Bool c1);

init_f init_glorot_uniform_dense;
init_f init_glorot_uniform_conv;
init_f init_bias;
init_f init_auto;

struct nlop_s;
_Complex float* init_auto_nlop_props(const struct nlop_s* op, unsigned int weight_index, _Complex float* src);
void init_nlop_weights(const struct nlop_s* op, unsigned long weight_flags, _Complex float* src);
