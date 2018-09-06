
#include "misc/mri.h"

extern void calc_sens(const long dims[DIMS], complex float* sens);

extern void calc_geo_phantom(const long dims[DIMS], complex float* out, _Bool ksp, int phtype, const long tstrs[DIMS], const _Complex float* traj);

extern void calc_phantom_noncart(const long dims[3], complex float* out, const complex float* traj);
extern void calc_geo_phantom_noncart(const long dims[3], complex float* out, const complex float* traj, int phtype);

extern void calc_phantom(const long dims[DIMS], _Complex float* out, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);
extern void calc_circ(const long dims[DIMS], _Complex float* img, _Bool d3, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);
extern void calc_ring(const long dims[DIMS], _Complex float* img, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);

extern void calc_moving_circ(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);
extern void calc_heart(const long dims[DIMS], _Complex float* out, _Bool ksp, const long tstrs[DIMS], const _Complex float* traj);

