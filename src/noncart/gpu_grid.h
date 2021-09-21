#include "misc/cppwrap.h"

struct grid_conf_s;
extern void cuda_grid(const struct grid_conf_s* conf, const _Complex float* traj, const long grid_dims[__VLA(4)], _Complex float* grid, const long ksp_dims[__VLA(4)], const _Complex float* src);
extern void cuda_grid2(const struct grid_conf_s* conf, const _Complex float* traj, const long grid_dims[__VLA(4)], _Complex float* grid, const long ksp_dims[__VLA(4)], const _Complex float* src);

#include "misc/cppwrap.h"


