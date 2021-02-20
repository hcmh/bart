
struct boundary_point_s {
	long offset;
	long index[3];
	_Complex float val;
	int dir[3];
};


_Complex float *calc_outward_normal(const long N, const long dims[N], const _Complex float *mask);
