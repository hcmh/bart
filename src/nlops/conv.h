#ifndef __PADD_ENUMS
#define __PADD_ENUMS
enum PADDING {PAD_VALID, PAD_SAME, PAD_CYCLIC, PAD_SYMMETRIC, PAD_REFLECT, PAD_CAUSAL};
#endif

extern struct nlop_s* nlop_convcorr_geom_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum PADDING conv_pad, _Bool conv, char transpc);
extern struct nlop_s* nlop_convcorr_fft_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum PADDING conv_pad, _Bool conv);