

extern const struct nlop_s* nlop_bias_create2(unsigned int N, const long dims[N],
					const long ostrs[N], const long istrs[N], const long bstrs[N], const complex float* bias);
extern const struct nlop_s* nlop_bias_create(unsigned int N, const long dims[N], const long bdims[N], const complex float* bias);


