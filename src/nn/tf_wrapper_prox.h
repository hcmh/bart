#include "tensorflow/c/tf_tensor.h"

struct nlop_s;
extern const struct nlop_s* nlop_tf_create(int nr_outputs, int nr_inputs, const char* path);
extern const struct TF_Tensor** get_input_tensor(struct nlop_s*);
extern const struct TF_Tensor** get_output_tensor(struct nlop_s*);
extern const struct TF_Tensor** get_grad_tensor(struct nlop_s*);