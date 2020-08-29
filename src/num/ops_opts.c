#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "num/multind.h"
#include "misc/debug.h"
#include "misc/misc.h"

#include "ops_opts.h"

struct op_options_s {

	unsigned int N;
	unsigned int NO;
	unsigned int NI;

	uint64_t io_flags;

	operator_option_flags_t* options;
};

struct op_property_s {

	unsigned int N;
	unsigned int NO;
	unsigned int NI;
	uint64_t io_flags;

	operator_property_flags_t* properties;
};


//static operator_option_flags_t op_options_get_flag(const struct op_options_s* x, unsigned int i, unsigned int j);
//static operator_option_flags_t op_options_get_io_flag(const struct op_options_s* x, unsigned int o, unsigned int i);
static void op_options_set_flag(struct op_options_s* x, unsigned int i, unsigned int j, operator_option_flags_t flags);
static void op_options_set_io_flag(struct op_options_s* x, unsigned int o, unsigned int i, operator_option_flags_t flags);

static operator_property_flags_t op_property_get_flag(const struct op_property_s* x, unsigned int i, unsigned int j);
static operator_property_flags_t op_property_get_io_flag(const struct op_property_s* x, unsigned int o, unsigned int i);
static void op_property_set_flag(const struct op_property_s* x, unsigned int i, unsigned int j, operator_property_flags_t flags);
static void op_property_set_io_flag(const struct op_property_s* x, unsigned int o, unsigned int i, operator_property_flags_t flags);



static unsigned int operator_index_to_io_index(uint64_t io_flags, unsigned int index, bool output)
{
	assert(output == (bool)MD_IS_SET(io_flags, index));
	unsigned int io_index = 0;

	for (unsigned int i = 0; i < index; i++)
		if (output == (bool)MD_IS_SET(io_flags, i))
			io_index++;

	return io_index;
}

static unsigned int operator_io_index_to_index(uint64_t io_flags, unsigned int io_index, bool output)
{
	unsigned int counter = 0;
	unsigned int index = 0;

	while ((counter < io_index) || (output != (bool)MD_IS_SET(io_flags, index))) {

		if (output == (bool)MD_IS_SET(io_flags, index))
			counter++;
		index++;
	}

	return index;
}

static struct op_options_s* op_options_create_internal(unsigned int N, uint64_t io_flags)
{
	PTR_ALLOC(struct op_options_s, data);
	data->N = N;

	io_flags = io_flags & (MD_BIT(N + 1) - 1);
	data->io_flags = io_flags;

	data->NO = bitcount(io_flags);
	data->NI = N - data->NO;

 	PTR_ALLOC(operator_option_flags_t[N * N], options);
	data->options = *PTR_PASS(options);

	for (unsigned int i = 0; i < N * N; i++)
		data->options[i] = 0;

	return PTR_PASS(data);
}

const struct op_options_s* op_options_create(unsigned int N, uint64_t io_flags, operator_option_flags_t options[N][N])
{
	struct op_options_s* result = op_options_create_internal(N, io_flags);

	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++) {

			if (NULL != options)
				op_options_set_flag(result, i, j, options[i][j]);
			else
				op_options_set_flag(result, i, j, 0);
		}

	return result;
}

const struct op_options_s* op_options_io_create(unsigned int NO, unsigned int NI, uint64_t io_flags, operator_option_flags_t options[NO][NI])
{
	struct op_options_s* result = op_options_create_internal(NO + NI, io_flags);

	for (unsigned int i = 0; i < NI; i++)
		for (unsigned int o = 0; o < NO; o++) {

			if (NULL != options)
				op_options_set_io_flag(result, o, i, options[o][i]);
			else
				op_options_set_io_flag(result, i, o, 0);
		}

	return result;
}

void op_options_free(const struct op_options_s* options)
{
	if (NULL == options)
		return;
	xfree(options->options);
	xfree(options);
}

static void op_options_set(struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(i < options->N);
	assert(j < options->N);
	options->options[options->N * i + j] = MD_SET(options->options[options->N * i + j], option);
}
static void op_options_set_sym(struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	op_options_set(options, i, j, option);
	op_options_set(options, j, i, option);
}

static void op_options_set_io(struct op_options_s* options, unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(o < (unsigned int)options->NO);
	assert(i < (unsigned int)options->NI);

	op_options_set_sym(options, operator_io_index_to_index(options->io_flags, o, true), operator_io_index_to_index(options->io_flags, i, false), option);
}
#if 0
static void op_options_set_oo(struct op_options_s* options, unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(o1 < (unsigned int)options->NO);
	assert(o2 < (unsigned int)options->NO);

	op_options_set_sym(options, operator_io_index_to_index(options->io_flags, o1, true), operator_io_index_to_index(options->io_flags, o2, true), option);
}

static void op_options_set_ii(struct op_options_s* options, unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(i1 < (unsigned int)options->NI);
	assert(i2 < (unsigned int)options->NI);

	op_options_set_sym(options, operator_io_index_to_index(options->io_flags, i1, false), operator_io_index_to_index(options->io_flags, i2, false), option);
}
#endif

static void op_options_clear(struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(i < options->N);
	assert(j < options->N);
	options->options[options->N * i + j] = MD_CLEAR(options->options[options->N * i + j], option);
}

static void op_options_clear_sym(struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	op_options_clear(options, i, j, option);
	op_options_clear(options, j, i, option);
}

static void op_options_clear_io(struct op_options_s* options, unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(o < (unsigned int)options->NO);
	assert(i < (unsigned int)options->NI);

	op_options_clear_sym(options, operator_io_index_to_index(options->io_flags, o, true), operator_io_index_to_index(options->io_flags, i, false), option);
}
#if 0
static void op_options_clear_oo(struct op_options_s* options, unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(o1 < (unsigned int)options->NO);
	assert(o2 < (unsigned int)options->NO);

	op_options_clear_sym(options, operator_io_index_to_index(options->io_flags, o1, true), operator_io_index_to_index(options->io_flags, o2, true), option);
}

static void op_options_clear_ii(struct op_options_s* options, unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(i1 < (unsigned int)options->NI);
	assert(i2 < (unsigned int)options->NI);

	op_options_clear_sym(options, operator_io_index_to_index(options->io_flags, i1, false), operator_io_index_to_index(options->io_flags, i2, false), option);
}


static operator_option_flags_t op_options_get_flag(const struct op_options_s* x, unsigned int i, unsigned int j)
{
	if (NULL == x)
		return 0;
	assert(i < x->N);
	assert(j < x->N);

	return x->options[x->N * i + j];
}

static operator_option_flags_t op_options_get_io_flag(const struct op_options_s* x, unsigned int o, unsigned int i)
{
	if (NULL == x)
		return 0;
	assert(i < x->NI);
	assert(o < x->NO);

	unsigned int ip = operator_io_index_to_index(x->io_flags, i, false);
	unsigned int op = operator_io_index_to_index(x->io_flags, o, true);
	return x->options[x->N * op + ip];
}
#endif
static void op_options_set_flag(struct op_options_s* x, unsigned int i, unsigned int j, operator_option_flags_t flags)
{
	assert(i < x->N);
	assert(j < x->N);

	x->options[x->N * i + j] = flags;
}

static void op_options_set_io_flag(struct op_options_s* x, unsigned int o, unsigned int i, operator_option_flags_t flags)
{
	assert(i < x->NI);
	assert(o < x->NO);

	unsigned int ip = operator_io_index_to_index(x->io_flags, i, false);
	unsigned int op = operator_io_index_to_index(x->io_flags, o, true);
	x->options[x->N * op + ip] = flags;
	x->options[x->N * ip + op] = flags;
}

bool op_options_is_set(const struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(i < options->N);
	assert(j < options->N);
	return (0 != MD_IS_SET(options->options[options->N * i + j], option));
}

bool op_options_is_set_sym(const struct op_options_s* options, unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	return op_options_is_set(options, i, j, option) && op_options_is_set(options, j, i, option);
}

bool op_options_is_set_io(const struct op_options_s* options, unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(o < (unsigned int)options->NO);
	assert(i < (unsigned int)options->NI);

	return op_options_is_set_sym(options, operator_io_index_to_index(options->io_flags, o, true), operator_io_index_to_index(options->io_flags, i, false), option);
}

bool op_options_is_set_oo(const struct op_options_s* options, unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(o1 < (unsigned int)options->NO);
	assert(o2 < (unsigned int)options->NO);

	return op_options_is_set_sym(options, operator_io_index_to_index(options->io_flags, o1, true), operator_io_index_to_index(options->io_flags, o2, true), option);
}

bool op_options_is_set_ii(const struct op_options_s* options, unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	assert(i1 < (unsigned int)options->NI);
	assert(i2 < (unsigned int)options->NI);

	return op_options_is_set_sym(options, operator_io_index_to_index(options->io_flags, i1, false), operator_io_index_to_index(options->io_flags, i2, false), option);
}

void print_operator_option_flags(const struct op_options_s* options)
{
	debug_printf(DP_INFO, "%d x %d runopts = \n", options->N, options->N);
	for (unsigned int i = 0; i < options->N; i++) {

		debug_printf(DP_INFO, "[");

		for (unsigned int j = 0; j < options->N; j++)
			debug_printf(DP_INFO, "%lu ", (unsigned long)options->options[options->N * i + j]);

		debug_printf(DP_INFO, "]\n");
	}
}

void print_operator_io_option_flags(const struct op_options_s* options)
{

	debug_printf(DP_INFO, "NO=%d x NI=%d io runopts = \n", options->NO, options->NI);
	for (unsigned int o = 0; o < options->NO; o++) {

		debug_printf(DP_INFO, "[");

		for (unsigned int i = 0; i < options->NI; i++)
			debug_printf(DP_INFO, "%lu ", (unsigned long)options->options[options->N * operator_io_index_to_index(options->io_flags, o, true) + operator_io_index_to_index(options->io_flags, i, false)]);

		debug_printf(DP_INFO, "]\n");
	}
}


const struct op_options_s* op_options_combine_create(const struct op_options_s* combine_options, unsigned int A, unsigned int off, uint64_t io_flags)
{
	struct op_options_s* result = op_options_create_internal(A, io_flags);

	for (unsigned int i = 0; i < A; i++)
		for (unsigned int j = 0; j < A; j++) {

			if (op_options_is_set(combine_options, off + i, off + j, OP_APP_NO_DER))
				op_options_set(result, i, j, OP_APP_NO_DER);
		}

	return result;
}

const struct op_options_s* op_options_dup_create(const struct op_options_s* dup_options, unsigned int a, unsigned int b, uint64_t io_flags)
{
	struct op_options_s* result = op_options_create_internal(dup_options->N + 1, io_flags);

	//only pass opts if only inputs are duplicated
	if (!MD_IS_SET(io_flags, a) && !MD_IS_SET(io_flags, b)) {

		unsigned int NO = result->NO;
		unsigned int NI = result->NI;

		unsigned int io_index_a = operator_index_to_io_index(io_flags, a, false);
		unsigned int io_index_b = operator_index_to_io_index(io_flags, b, false);

		for (unsigned int o = 0; o < NO; o++) {

			for(unsigned int i = 0, ip = 0; i < NI; i++) {

				if (i == io_index_b)
					continue;
				if (op_options_is_set_io(dup_options, o, ip, OP_APP_NO_DER))
					op_options_set_io(result, o, i, OP_APP_NO_DER);
				ip++;
			}

			if (op_options_is_set_io(result, o, io_index_a, OP_APP_NO_DER))
				op_options_set_io(result, o, io_index_b, OP_APP_NO_DER);
		}
	}

	return result;
}

const struct op_options_s* op_options_link_create(const struct op_options_s* link_options, unsigned int a, unsigned int b, uint64_t io_flags, const struct op_property_s* prop)
{
	unsigned int out_ind = operator_index_to_io_index(io_flags, b, true);
	unsigned int in_ind = operator_index_to_io_index(io_flags, a, false);

	struct op_options_s* result = op_options_create_internal(link_options->N + 2, io_flags);

	unsigned int NO = result->NO;
	unsigned int NI = result->NI;

	for (unsigned int ip = 0; ip < NI; ip++)
		for (unsigned int op = 0; op < NO; op++)
			op_options_set_io(result, op, ip, OP_APP_NO_DER);

	/*
	* Select needed derivatives (loop over i, j)
	* dg_o/dx_i = df_op/dx_ip + df_op/dx_in * df_out/dx_ip
	*/
	for (unsigned int o = 0, op = -1; o < NO - 1; o++) {

		op += (o == out_ind) ? 2 : 1;

		for (unsigned int i = 0, ip = -1; i < NI - 1; i++) {

			ip += (i == in_ind) ? 2 : 1;

			if (!op_options_is_set_io(link_options, o, i, OP_APP_NO_DER)) {

				op_options_clear_io(result, op, ip, OP_APP_NO_DER);
				if (!(op_property_is_set_io(prop, out_ind, ip, OP_PROP_INDEPENDENT)))
					op_options_clear_io(result, op, in_ind, OP_APP_NO_DER);
				if (!(op_property_is_set_io(prop, op, in_ind, OP_PROP_INDEPENDENT)))
					op_options_clear_io(result, out_ind, ip, OP_APP_NO_DER);
			}
		}
	}

	return result;
}

const struct op_options_s* op_options_permute_create(const struct op_options_s* permute_options, unsigned int N, const int perm[N])
{

	uint64_t io_flags = 0;
	for (unsigned int i = 0; i < N; i++)
		if (MD_IS_SET(permute_options->io_flags, i))
			io_flags = MD_SET(io_flags, perm[i]);

	struct op_options_s* result = op_options_create_internal(permute_options->N, io_flags);

	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++) {

			if (op_options_is_set(permute_options, i, j, OP_APP_NO_DER))
				op_options_set(result, perm[i], perm[j], OP_APP_NO_DER);
		}

	return result;
}

const struct op_options_s* op_options_select_der_create(unsigned int NO, unsigned int NI, uint32_t out_der_flag, uint32_t in_der_flag)
{
	struct op_options_s* result = op_options_create_internal(NO + NI, MD_BIT(NO) - 1);
	for (unsigned int o = 0; o < NO; o++)
		for (unsigned int i = 0; i < NI; i++)
			if (!(MD_IS_SET(out_der_flag, o) && MD_IS_SET(in_der_flag, i)))
				op_options_set_io(result, o, i, OP_APP_NO_DER);

	return result;
}

static struct op_property_s* op_property_create_internal(unsigned int N, uint64_t io_flags)
{
	PTR_ALLOC(struct op_property_s, data);
	data->N = N;

	io_flags = io_flags & (MD_BIT(N + 1) - 1);
	data->io_flags = io_flags;

	data->NO = bitcount(io_flags);
	data->NI = N - data->NO;

	PTR_ALLOC(operator_property_flags_t[N * N], properties);
	data->properties = *PTR_PASS(properties);

	for (unsigned int i = 0; i < N * N; i++)
		data->properties[i] = 0;

	return PTR_PASS(data);
}

const struct op_property_s* op_property_create(unsigned int N, uint64_t io_flags, operator_property_flags_t properties[N][N])
{
	struct op_property_s* result = op_property_create_internal(N, io_flags);

	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++) {

			if (NULL != properties)
				op_property_set_flag(result, i, j, properties[i][j]);
			else
				op_property_set_flag(result, i, j, 0);
		}

	return result;
}

const struct op_property_s* op_property_io_create(unsigned int NO, unsigned int NI, uint64_t io_flags, operator_option_flags_t properties[NO][NI])
{
	struct op_property_s* result = op_property_create_internal(NO + NI, io_flags);

	for (unsigned int i = 0; i < NI; i++)
		for (unsigned int o = 0; o < NO; o++) {

			if (NULL != properties)
				op_property_set_io_flag(result, o, i, properties[o][i]);
			else
				op_property_set_io_flag(result, i, o, 0);
		}

	return result;
}


void op_property_free(const struct op_property_s* options)
{
	if (NULL == options)
		return;
	xfree(options->properties);
	xfree(options);
}

static operator_property_flags_t op_property_get_flag(const struct op_property_s* x, unsigned int i, unsigned int j)
{
	if (NULL == x)
		return 0;
	assert(i < x->N);
	assert(j < x->N);

	return x->properties[x->N * i + j];
}

static operator_property_flags_t op_property_get_io_flag(const struct op_property_s* x, unsigned int o, unsigned int i)
{
	if (NULL == x)
		return 0;
	assert(i < x->NI);
	assert(o < x->NO);

	unsigned int ip = operator_io_index_to_index(x->io_flags, i, false);
	unsigned int op = operator_io_index_to_index(x->io_flags, o, true);
	return x->properties[x->N * op + ip];
}


static void op_property_set_flag(const struct op_property_s* x, unsigned int i, unsigned int j, operator_property_flags_t flags)
{
	assert(i < x->N);
	assert(j < x->N);

	x->properties[x->N * i + j] = flags;
}

static void op_property_set_io_flag(const struct op_property_s* x, unsigned int o, unsigned int i, operator_property_flags_t flags)
{
	assert(i < x->NI);
	assert(o < x->NO);

	unsigned int ip = operator_io_index_to_index(x->io_flags, i, false);
	unsigned int op = operator_io_index_to_index(x->io_flags, o, true);
	x->properties[x->N * op + ip] = flags;
	x->properties[x->N * ip + op] = flags;
}



const struct op_property_s* op_property_combine_create(unsigned int N, const struct op_property_s* properties[N])
{
	unsigned int A = 0;
	uint64_t io_flags = 0;
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < properties[i]->N; j++) {

			io_flags |= ((properties[i]->io_flags >> j) & 1) << A;
			A++;
		}
	}

	struct op_property_s* result = op_property_create_internal(A, io_flags);
	for (unsigned int i = 0; i < A; i++)
		for (unsigned int j = 0; j < A; j++)
			op_property_set_flag(result, i, j, MD_BIT(OP_PROP_INDEPENDENT));

	unsigned int offset = 0;
	for (unsigned int k = 0; k < N; k++){

		for (unsigned int i = 0; i < properties[k]->N; i++)
			for (unsigned int j = 0; j < properties[k]->N; j++)
				op_property_set_flag(result, offset + i, offset + j, op_property_get_flag(properties[k], i, j));

		offset += properties[k]->N;
	}

	return result;
}

const struct op_property_s* op_property_dup_create(const struct op_property_s* dup_prop, unsigned int a, unsigned int b, uint64_t io_flags)
{
	struct op_property_s* result = op_property_create_internal(dup_prop->N - 1, io_flags);

	for (unsigned int i = 0, ip = 0; i < dup_prop->N; i++) {

		if (b == i)
			continue;

		for (unsigned int j = 0, jp = 0; j < dup_prop->N; j++) {

			if (b == j)
				continue;

			op_property_set_flag(result, ip, jp, op_property_get_flag(dup_prop, i, j));

			if (a == i)
				op_property_set_flag(result, ip, jp, op_property_get_flag(result, ip, jp) & op_property_get_flag(dup_prop, b, j));
			if (a == j)
				op_property_set_flag(result, ip, jp, op_property_get_flag(result, ip, jp) & op_property_get_flag(dup_prop, i, b));
			if ((a == i) && (a == j))
				op_property_set_flag(result, ip, jp, op_property_get_flag(result, ip, jp) & op_property_get_flag(dup_prop, b, b));
			jp++;
		}
		ip++;
	}

	return result;
}

const struct op_property_s* op_property_link_create(const struct op_property_s* link_prop, unsigned int o, unsigned int i, uint64_t io_flags)
{
	struct op_property_s* result = op_property_create_internal(link_prop->N - 2, io_flags);

	for (unsigned int k = 0, kp = 0; k < link_prop->N; k++) {

		if ((k == i) || (k == o))
			continue;

		for (unsigned int l = 0, lp = 0; l < link_prop->N; l++) {

			if ((l == i) || (l == o))
				continue;

			op_property_set_flag(result, kp, lp, op_property_get_flag(link_prop, k, l));

			if (    MD_IS_SET(io_flags, kp)
			    && !MD_IS_SET(io_flags, lp)
			    && !op_property_is_set(link_prop, k, i, OP_PROP_INDEPENDENT)
			    && !op_property_is_set(link_prop, o, l, OP_PROP_INDEPENDENT) ) {

				op_property_set_flag(result, kp, lp, (~MD_BIT(OP_PROP_INDEPENDENT)) & op_property_get_flag(result, kp, lp));
			}

			if (    MD_IS_SET(io_flags, lp)
			    && !MD_IS_SET(io_flags, kp)
			    && !op_property_is_set(link_prop, l, i, OP_PROP_INDEPENDENT)
			    && !op_property_is_set(link_prop, o, k, OP_PROP_INDEPENDENT) ) {

				op_property_set_flag(result, kp, lp, (~MD_BIT(OP_PROP_INDEPENDENT)) & op_property_get_flag(result, kp, lp));
			}

			lp++;
		}
		kp++;
	}

	return result;
}

const struct op_property_s* op_property_permute_create(const struct op_property_s* permute_prop, unsigned int N, const int perm[N], uint64_t io_flags)
{
	struct op_property_s* result = op_property_create_internal(permute_prop->N, io_flags);

	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++)
			op_property_set_flag(result, i, j, op_property_get_flag(permute_prop, perm[i], perm[j]));

	return result;
}

const struct op_property_s* op_property_clone(const struct op_property_s* prop)
{
	struct op_property_s* result = op_property_create_internal(prop->N, prop->io_flags);

	for (unsigned int i = 0; i < prop->N; i++)
		for (unsigned int j = 0; j < prop->N; j++)
			op_property_set_flag(result, i, j, op_property_get_flag(prop, i, j));

	return result;
}




bool op_property_is_set(const struct op_property_s* x, unsigned int i, unsigned int j, enum OPERATOR_IO_PROP_FLAGS_INDEX prop)
{
	return 0 != MD_IS_SET(op_property_get_flag(x, i, j), prop);
}

bool op_property_is_set_io(const struct op_property_s* x, unsigned int o, unsigned int i, enum OPERATOR_IO_PROP_FLAGS_INDEX prop)
{
	return 0 != MD_IS_SET(op_property_get_io_flag(x, o, i), prop);
}

void op_property_io_print(int debug_level, const struct op_property_s* x)
{
	debug_printf(debug_level, "%d x %d operator properties= \n", x->NO, x->NI);
	for (unsigned int o = 0; o < x->NO; o++) {

		debug_printf(debug_level, "[");

		for (unsigned int i = 0; i < x->NI; i++)
			debug_printf(debug_level, "%lu ", (unsigned long)op_property_get_io_flag(x, o, i));

		debug_printf(debug_level, "]\n");
	}
}
