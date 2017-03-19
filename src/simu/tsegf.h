
_Complex double tse_gf(_Complex double z, double k1, double k2, double cosa);
_Complex double tse_Dgf_k1(_Complex double z, double k1, double k2, double cosa);
_Complex double tse_Dgf_k2(_Complex double z, double k1, double k2, double cosa);
_Complex double tse_Dgf_ca(_Complex double z, double k1, double k2, double cosa);

extern void tse(unsigned int N, _Complex float out[N], unsigned int M, const float in[4]);
extern void tse_der(unsigned int N, _Complex float out[N], unsigned int M,
	const float in[4], const float Din[4]);
extern void tse_adj(unsigned int N, float out[4], unsigned int M,
	float in[4], _Complex float inb[N]);

