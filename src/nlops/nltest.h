
struct nlop_s;
extern float nlop_test_derivative(const struct nlop_s* op);

extern float nlop_test_adj_derivatives(const struct nlop_s* op, _Bool real);
extern float nlop_test_derivatives(const struct nlop_s* op);

extern float compare_gpu(const struct nlop_s* cpu_op, const struct nlop_s* gpu_op);
