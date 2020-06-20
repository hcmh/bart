#include "num/ops_opts.h"

struct nlop_s;

extern operator_prop_flags_t nlop_get_ii_props(const struct nlop_s* op, unsigned int i1, unsigned int i2);
extern operator_prop_flags_t nlop_get_oo_props(const struct nlop_s* op, unsigned int o1, unsigned int o2);
extern operator_prop_flags_t nlop_get_oi_props(const struct nlop_s* op, unsigned int o, unsigned int i);

extern const struct nlop_s* nlop_set_nn_in_type_F(const struct nlop_s* op, unsigned int i, enum OPERATOR_IO_PROP_FLAGS_INDEX type);
extern operator_prop_flags_t nlop_get_nn_in_type(const struct nlop_s* op, unsigned int i);
extern const struct nlop_s* nlop_set_batchnorm_F(const struct nlop_s* op, unsigned int o, unsigned int i);