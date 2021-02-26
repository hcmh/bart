#ifndef N_boundary_point_s
#define N_boundary_point_s 4
struct boundary_point_s {
	long index[N_boundary_point_s];
	_Complex float val;
	int dir[N_boundary_point_s];
};
#endif

void calc_outward_normal(const long N, const long grad_dims[N], _Complex float *grad, const long grad_dim, const long dims[N], const _Complex float *mask);
long calc_boundary_points(const long N, const long dims[N], struct boundary_point_s *boundary, const long grad_dim, const _Complex float *normal, const _Complex float *values);

void clear_mask_forward(const long N, const long dims[N], _Complex float *out, const long n_points, const struct boundary_point_s *boundary, const _Complex float *mask);

void shrink_wrap(const long N, const long dims[N], _Complex float *out, const long n_points, const struct boundary_point_s *boundary, const _Complex float *mask);
