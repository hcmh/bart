#include "simu/boundary.h"

struct sparse_diag_s *sd_laplace_create(long N, const long dims[N]);

void laplace_dirichlet(struct sparse_diag_s *mat, const long N, const long dims[N], const long n_points, struct boundary_point_s points[n_points]);

void laplace_neumann(struct sparse_diag_s *mat, const long N, const long dims[N], const long n_points, const struct boundary_point_s boundary[n_points]);

void laplace_neumann_update_rhs(const long N, const long dims[N], _Complex float *rhs, const long n_points, const struct boundary_point_s boundary[n_points]);

struct linop_s *linop_laplace_neumann_create(const long N, const long dims[N], const _Complex float *mask, const long n_points, const struct boundary_point_s boundary[n_points]);
