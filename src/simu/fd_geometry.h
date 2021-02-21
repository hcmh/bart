#define N_boundary_point_s 3

struct boundary_point_s {
	long index[N_boundary_point_s];
	_Complex float val;
	float dir[N_boundary_point_s];
};

void calc_outward_normal(const long N, const long grad_dims[N], _Complex float *grad, const long grad_dim, const long dims[N], const _Complex float *mask);
long calc_boundary_points(const long N, const long dims[N], struct boundary_point_s *boundary, const long grad_dim, const _Complex float *normal);
