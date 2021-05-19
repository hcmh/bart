/*
 * Authors:
 * 2020 Philip Schaten <philip.schaten@med.uni-goettingen.de>
 *
 */

#ifndef __BC_ENUMS
#define __BC_ENUMS
enum BOUNDARY_CONDITION { BC_PERIODIC,
			  BC_ZERO,
			  BC_SAME };
#endif

#include "iter/monitor.h"
#include "linops/linop.h"

void linop_leray_calc_rhs(const linop_data_t *_data, _Complex float *y, const _Complex float *src);
void linop_leray_calc_projection(const linop_data_t *_data, _Complex float *dst, const _Complex float *tmp);
struct linop_s *linop_leray_create(const long N, const long dims[N], long vec_dim, const int iter, const float lambda, const _Complex float *mask, struct iter_monitor_s *mon);
const struct operator_p_s *prox_indicator_create(const struct linop_s *op);
