#ifndef N_boundary_point_s
#define N_boundary_point_s 4
struct boundary_point_s {
	long index[N_boundary_point_s];
	_Complex float val;
	int dir[N_boundary_point_s];
};
#endif
