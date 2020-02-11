enum CONV_PAD {PADDING_VALID, PADDING_SAME, PADDING_CYCLIC, PADDING_CAUSAL};

#ifndef TIMER
#define TIMER
#define START_TIMER static double time = 0.; static double count = 0.; time -= timestamp();
#define PRINT_TIMER(name) { time += timestamp(); count += 1.; debug_printf(DP_DEBUG1, "%.0f %s\tapplied in %3.6f seconds\n", count, name, time); }
#endif

struct nlop_s;
extern struct nlop_s* nlop_conv_create(int N, unsigned int flags, const long odims[N], const long idims1[N], const long idims2[N]);
extern struct nlop_s* nlop_conv_geom_create(int N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum CONV_PAD conv_pad);
extern struct nlop_s* nlop_conv_fft_create(int N, unsigned int flags, const long odims[N], const long idims[N], const long kdims[N], enum CONV_PAD conv_pad);
