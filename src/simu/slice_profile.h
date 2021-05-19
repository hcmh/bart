
#include <complex.h>

struct simdata_pulse;
extern void estimate_slice_profile(unsigned int N, const long dims[N], complex float* out, struct simdata_pulse* pulse);
