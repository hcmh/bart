#include "simu/boundary.h"

void calc_outward_normal(const long N, const long grad_dims[N], _Complex float *grad, const long grad_dim, const long dims[N], const _Complex float *mask);
long calc_boundary_points(const long N, const long dims[N], struct boundary_point_s *boundary, const long grad_dim, const _Complex float *normal, const _Complex float *values);

void fill_holes(const long N, const long grad_dims[N], const long grad_dim, const long dims[N], _Complex float* out, const _Complex float *mask);
void clear_mask_forward(const long N, const long dims[N], _Complex float *out, const long n_points, const struct boundary_point_s *boundary, const _Complex float *mask);

void shrink_wrap(const long N, const long dims[N], _Complex float *dst, const long n_points, const struct boundary_point_s *boundary, const _Complex float *src);

void neumann_set_boundary(const long N, const long dims[N], const long vecdim, _Complex float *dst, const long n_points, const struct boundary_point_s *boundary, const _Complex float *src);
