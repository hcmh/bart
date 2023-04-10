 
#ifndef __PROX2_H
#define __PROX2_H

#include "misc/cppwrap.h"

struct operator_p_s;
struct linop_s;
struct nlop_s;
struct dp_conf;

extern const struct operator_p_s* prox_normaleq_create(const struct linop_s* op, const _Complex float* y);
extern const struct operator_p_s* prox_lineq_create(const struct linop_s* op, const _Complex float* y);
extern const struct operator_p_s* prox_nlgrad_create(const struct nlop_s* op, int steps, float stepsize, float lambda);
extern const struct operator_p_s* prox_nlgrad_create2(const struct nlop_s* op, int steps, float stepsize, float lambda);
extern const struct operator_p_s* prox_nl_dp_grad_create(const struct nlop_s* op, unsigned int iter, float lambda);
extern const struct operator_p_s* op_p_auto_resize(const struct operator_p_s* op, int N, const long resize_dims[__VLA(N)], const long img_dims[__VLA(N)]);

enum norm { NORM_MAX, NORM_L2 };
extern const struct operator_p_s* op_p_auto_normalize(const struct operator_p_s* op, long flags, enum norm norm);
extern const struct operator_p_s* op_p_auto_normalize2(const struct operator_p_s* op, int N, long flags, enum norm norm, const long img_dims[__VLA(N)]);
extern const struct operator_p_s* op_p_conjugate(const struct operator_p_s* op, const struct linop_s* lop);

#include "misc/cppwrap.h"

#endif

