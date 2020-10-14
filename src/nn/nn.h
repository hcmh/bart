
#ifndef NN_H
#define NN_H

#include "misc/debug.h"

#include "iter/italgos.h"

struct nlop_s;
struct initializer_s;

struct nn_s {

	const struct nlop_s* network;

	const char** out_names;
	const char**  in_names;

	const struct initializer_s** initializers;

	enum IN_TYPE* in_types;
	enum OUT_TYPE* out_types;
};

typedef const struct nn_s* nn_t;

extern nn_t nn_from_nlop(const struct nlop_s* op);
extern nn_t nn_from_nlop_F(const struct nlop_s* op);
extern void nn_free(nn_t op);

extern const struct nlop_s* nn_get_nlop(nn_t op);

extern nn_t nn_clone(nn_t op);
extern void nn_clone_arg_i_from_i(nn_t nn1, unsigned int i1, nn_t nn2, unsigned int i2);
extern void nn_clone_arg_o_from_o(nn_t nn1, unsigned int o1, nn_t nn2, unsigned int o2);
extern void nn_clone_args(nn_t dst, nn_t src);

extern unsigned int nn_get_nr_named_in_args(nn_t op);
extern unsigned int nn_get_nr_named_out_args(nn_t op);
extern unsigned int nn_get_nr_unnamed_in_args(nn_t op);
extern unsigned int nn_get_nr_unnamed_out_args(nn_t op);
extern unsigned int nn_get_nr_in_args(nn_t op);
extern unsigned int nn_get_nr_out_args(nn_t op);

extern int nn_get_out_arg_index(nn_t op, int o, const char* oname);
extern int nn_get_in_arg_index(nn_t op, int i, const char* iname);

extern _Bool nn_is_num_in_index(nn_t op, unsigned int i);
extern _Bool nn_is_num_out_index(nn_t op, unsigned int o);

extern void nn_get_in_args_names(nn_t op, int nII, const char* inames[nII]);
extern void nn_get_out_args_names(nn_t op, int nOO, const char* onames[nOO]);

extern const char* nn_get_in_name_from_arg_index(nn_t op, int i);
extern const char* nn_get_out_name_from_arg_index(nn_t op, int o);

extern int nn_get_in_index_from_arg_index(nn_t op, int i);
extern int nn_get_out_index_from_arg_index(nn_t op, int o);

extern _Bool nn_is_name_in_in_args(nn_t op, const char* name);
extern _Bool nn_is_name_in_out_args(nn_t op, const char* name);

extern nn_t nn_set_input_name_F(nn_t op, int i, const char* iname);
extern nn_t nn_set_output_name_F(nn_t op, int o, const char* oname);
extern nn_t nn_rename_input_F(nn_t op, const char* nname, const char* oname);
extern nn_t nn_rename_output_F(nn_t op, const char* nname, const char* oname);
extern nn_t nn_unset_input_name_F(nn_t op, const char* iname);
extern nn_t nn_unset_output_name_F(nn_t op, const char* oname);

extern nn_t nn_set_initializer_F(nn_t op, int i, const char* iname, const struct initializer_s* ini);
extern nn_t nn_set_in_type_F(nn_t op, int i, const char* iname, enum IN_TYPE in_type);
extern nn_t nn_set_out_type_F(nn_t op, int o, const char* oname, enum OUT_TYPE out_type);

extern const char** nn_get_out_names(nn_t op);
extern const char** nn_get_in_names(nn_t op);

extern int nn_get_nr_weights(nn_t op);
extern void nn_get_in_types(nn_t op, unsigned int N, enum IN_TYPE in_types[N]);
extern void nn_get_out_types(nn_t op, unsigned int N, enum OUT_TYPE out_types[N]);

extern const struct iovec_s* nn_generic_domain(nn_t op, int i, const char* iname);
extern const struct iovec_s* nn_generic_codomain(nn_t op, int o, const char* oname);

extern void nn_debug(enum debug_levels dl, nn_t x);

#endif