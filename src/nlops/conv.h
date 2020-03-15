#ifndef NLOPS_CONV_H
#define NLOPS_CONV_H

enum CONV_PAD {PADDING_VALID, PADDING_SAME, PADDING_CYCLIC, PADDING_CAUSAL};

extern struct nlop_s* nlop_convcorr_geom_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum CONV_PAD conv_pad, _Bool conv);
extern struct nlop_s* nlop_convcorr_fft_create(long N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum CONV_PAD conv_pad, _Bool conv);

#endif