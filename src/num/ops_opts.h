#include <stdint.h>

#ifndef __OPERATOR_OPTS_H
#define __OPERATOR_OPTS_H



enum OPERATOR_RUN_OPT_FLAGS_INDEX {	OP_APP_NO_DER, 		// nlops do not need to store information for derivative
					OP_APP_CLEAR_DER, 	// delete information for derivative
					};
enum OPERATOR_IO_PROP_FLAGS_INDEX {	OP_PROP_ATOMIC, // in/out puts belong to the same operator on the lowest level
					OP_PROP_R_LIN,	// operator is linear over the real numbers
					OP_PROP_C_LIN,	// operator is linear over the complex numbers
					OP_PROP_HOLOMORPHIC,	// function is holomorphic, i.e. the derivative is c linear
					OP_PROP_INDEPENDENT 	// in/out are (mathematically) independent, i.e. derivatives vanish
					};

typedef uint32_t operator_option_flags_t;
typedef uint32_t operator_property_flags_t;

struct op_options_s;
struct op_property_s;

extern const struct op_options_s* op_options_create(unsigned int N, const _Bool io_flags[N], operator_option_flags_t options[N][N]);
extern const struct op_options_s* op_options_io_create(unsigned int NO, unsigned int NI, const _Bool io_flags[NO + NI], operator_option_flags_t options[NO][NI]);
extern void op_options_free(const struct op_options_s* options);
extern unsigned int op_options_get_N(const struct op_options_s* option);
extern _Bool op_options_check_io_flags(const struct op_options_s* option, unsigned int N, const _Bool io_flags[N]);
extern const struct op_options_s* op_options_clone(const struct op_options_s* prop);

extern _Bool op_options_is_set(const struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX optionion);
extern _Bool op_options_is_set_sym(const struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX optionion);
extern _Bool op_options_is_set_io(const struct op_options_s* options, unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX optionion);
extern _Bool op_options_is_set_oo(const struct op_options_s* options, unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX optionion);
extern _Bool op_options_is_set_ii(const struct op_options_s* options, unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX optionion);

extern const struct op_options_s* op_options_combine_create(const struct op_options_s* combine_options, unsigned int A, unsigned int off, const _Bool io_flags[A]);
extern const struct op_options_s* op_options_dup_create(const struct op_options_s* dup_options, unsigned int a, unsigned int b, unsigned int N, const _Bool io_flags[N]);
extern const struct op_options_s* op_options_link_create(const struct op_options_s* link_options, unsigned int a, unsigned int b, const struct op_property_s* prop);
extern const struct op_options_s* op_options_permute_create(const struct op_options_s* permute_options, unsigned int N, const int perm[N]);

extern const struct op_options_s* op_options_select_der_create(unsigned int NO, unsigned int NI, _Bool out_der_flag[NO], _Bool in_der_flag[NI]);
extern const struct op_options_s* op_options_clear_der_create(unsigned int NO, unsigned int NI);

extern void print_operator_option_flags(const struct op_options_s* options);
extern void print_operator_io_option_flags(const struct op_options_s* options);

extern const struct op_property_s* op_property_create(unsigned int N, const _Bool io_flags[N], operator_property_flags_t flags[N][N]);
extern const struct op_property_s* op_property_io_create(unsigned int NO, unsigned int NI, const _Bool io_flags[NO + NI], operator_property_flags_t flags[NO][NI]);
extern void op_property_free(const struct op_property_s* x);
extern unsigned int op_property_get_N(const struct op_property_s* prop);
extern _Bool op_property_check_io_flags(const struct op_property_s* prop, unsigned int N, const _Bool io_flags[N]);

extern const struct op_property_s* op_property_combine_create(unsigned int N, const struct op_property_s* properties[N]);
extern const struct op_property_s* op_property_dup_create(const struct op_property_s* dup_prop, unsigned int a, unsigned int b);
extern const struct op_property_s* op_property_link_create(const struct op_property_s* link_prop, unsigned int o, unsigned int i);
extern const struct op_property_s* op_property_permute_create(const struct op_property_s* permute_prop, unsigned int N, const int perm[N], const _Bool io_flags[N]);
extern const struct op_property_s* op_property_clone(const struct op_property_s* prop);

extern _Bool op_property_is_set(const struct op_property_s* x, unsigned int i, unsigned int j, enum OPERATOR_IO_PROP_FLAGS_INDEX property);
extern _Bool op_property_is_set_io(const struct op_property_s* x, unsigned int o, unsigned int i, enum OPERATOR_IO_PROP_FLAGS_INDEX property);

extern operator_property_flags_t op_property_get_flag(const struct op_property_s* x, unsigned int i, unsigned int j);
extern operator_property_flags_t op_property_get_io_flag(const struct op_property_s* x, unsigned int o, unsigned int i);

extern void op_property_io_print(int debug_level, const struct op_property_s* x);

#endif //OPERATOR_OPTS_H