
extern const struct operator_s* operator_relu_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N]);
extern const struct operator_s* operator_relu_create(unsigned int N, const long dims[N]);

extern void simple_dcnn(const long dims[6], const long krn_dims[6], const complex float* krn, const long bias_dims[6], const complex float* bias, complex float* out, const complex float* in);

extern const struct operator_p_s* prox_simple_dcnn_create(unsigned int N,
	const long dims[6], const long krn_dims[6], const complex float* krn, const long bias_dims[6], const complex float* bias, float alpha);
