#include <stdint.h>

#ifndef __OPERATOR_OPTS_H
#define __OPERATOR_OPTS_H
enum OPERATOR_RUN_OPT_FLAGS_INDEX {	OP_APP_NO_DER, 	// nlops do not need to store information for derivative
					};
enum OPERATOR_IO_PROP_FLAGS_INDEX {	OP_PROP_ATOMIC, // in/out puts belong to the same operator on the lowest level
					OP_PROP_R_LIN,	// operator is linear over the real numbers
					OP_PROP_C_LIN,	// operator is linear over the complex numbers
					OP_PROP_HOLOMORPHIC,	// function is holomorphic, i.e. the derivative is c linear
					OP_PROP_INDEPENDENT // in/out are (mathematically) independent, i.e. derivatives vanish
					};

typedef uint32_t operator_run_opt_flags_t;
typedef uint32_t operator_prop_flags_t;

struct opprop_s {

	int N;
	int NO;
	int NI;
	unsigned int io_flags;

	const operator_prop_flags_t* flags;
};

extern void print_operator_run_flags(int N, operator_run_opt_flags_t run_opts[N][N]);
extern void print_operator_io_run_flags(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI]);
extern void operator_run_opts_to_io_run_opts(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t io_run_opts[NO][NI], operator_run_opt_flags_t run_opts[NO + NI][NO + NI]);
extern void operator_io_run_opts_to_run_opts(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], operator_run_opt_flags_t io_run_opts[NO][NI]);

unsigned int operator_index_to_io_index(unsigned int io_flags, unsigned int index, _Bool output);
unsigned int operator_io_index_to_index(unsigned int io_flags, unsigned int io_index, _Bool output);

void operator_init_run_opts(int N, operator_run_opt_flags_t run_opts[N][N]);

void operator_set_run_opt(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_set_run_opt_sym(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_set_oi_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_set_oo_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_set_ii_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);

void operator_unset_run_opt(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_unset_run_opt_sym(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_unset_oi_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_unset_oo_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
void operator_unset_ii_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);

_Bool operator_get_run_opt(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
_Bool operator_get_run_opt_sym(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
_Bool operator_get_oi_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
_Bool operator_get_oo_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);
_Bool operator_get_ii_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option);

const struct opprop_s* opprop_create(unsigned int N, unsigned int io_flags, operator_prop_flags_t flags[N][N]);
const struct opprop_s* opprop_io_create(unsigned int NO, unsigned int NI, unsigned int io_flags, operator_prop_flags_t flags[NO][NI]);
void opprop_free(const struct opprop_s* x);
operator_prop_flags_t opprop_get(const struct opprop_s* x, unsigned int i, unsigned int j);
operator_prop_flags_t opprop_io_get(const struct opprop_s* x, unsigned int o, unsigned int i);
_Bool opprop_isset(const struct opprop_s* x, unsigned int i, unsigned int j, enum OPERATOR_IO_PROP_FLAGS_INDEX prop);
_Bool opprop_io_isset(const struct opprop_s* x, unsigned int o, unsigned int i, enum OPERATOR_IO_PROP_FLAGS_INDEX prop);
void opprop_io_print(int debug_level, const struct opprop_s* x);

#endif //OPERATOR_OPTS_H