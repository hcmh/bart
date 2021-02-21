
struct boundary_point_s {
	long offset;
	long index[3];
	_Complex float val;
	int dir[3];
};


void calc_outward_normal(const long N, const long grad_dims[N], _Complex float *grad, const long grad_dim, const long dims[N], const _Complex float *mask);
