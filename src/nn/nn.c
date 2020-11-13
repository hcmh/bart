
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/types.h"

#include "nn/initializer.h"
#include "nn/chain.h"
#include "num/multind.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "iter/italgos.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/checkpointing.h"

#include "nn.h"


nn_t nn_from_nlop(const struct nlop_s* op)
{
	PTR_ALLOC(struct nn_s, nn);

	uint NO = nlop_get_nr_out_args(op);
	uint NI = nlop_get_nr_in_args(op);

	PTR_ALLOC(const char*[NO], out_names);
	PTR_ALLOC(const char*[NI], in_names);

	PTR_ALLOC(const struct initializer_s*[NI], initializers);
	PTR_ALLOC(enum IN_TYPE[NI], in_types);
	PTR_ALLOC(enum OUT_TYPE[NO], out_types);

	for (uint i = 0; i < NI; i++) {

		(*in_names)[i] = NULL;
		(*initializers)[i] = NULL;
		(*in_types)[i] = IN_UNDEFINED;
	}

	for (uint o = 0; o < NO; o++) {

		(*out_names)[o] = NULL;
		(*out_types)[o] = OUT_UNDEFINED;
	}

	nn->in_names = *PTR_PASS(in_names);
	nn->out_names = *PTR_PASS(out_names);

	nn->initializers = *PTR_PASS(initializers);
	nn->in_types = *PTR_PASS(in_types);
	nn->out_types = *PTR_PASS(out_types);

	nn->network = nlop_clone(op);

	return PTR_PASS(nn);
}

void nn_free(nn_t op)
{
	uint II = nn_get_nr_in_args(op);
	uint OO = nn_get_nr_out_args(op);

	for (uint i = 0; i < II; i++){

		xfree(op->in_names[i]);
		initializer_free(op->initializers[i]);
	}
	for (uint o = 0; o < OO; o++)
		xfree(op->out_names[o]);

	xfree(op->in_names);
	xfree(op->out_names);

	xfree(op->initializers);
	xfree(op->in_types);
	xfree(op->out_types);

	nlop_free(op->network);

	xfree(op);
}

nn_t nn_from_nlop_F(const struct nlop_s* op)
{
	auto result = nn_from_nlop(op);
	nlop_free(op);
	return result;
}

const struct nlop_s* nn_get_nlop(nn_t op)
{
	return op->network;
}

void nn_clone_arg_i_from_i(nn_t nn1, uint i1, nn_t nn2, uint i2)
{
	if (NULL != nn1->in_names[i1])
		xfree(nn1->in_names[i1]);

	if (NULL != nn2->in_names[i2]) {

		PTR_ALLOC(char[strlen(nn2->in_names[i2]) + 1], name);
		strcpy(*name, nn2->in_names[i2]);
		nn1->in_names[i1] = *PTR_PASS(name);
	} else {

		nn1->in_names[i1] = NULL;
	}

	initializer_free(nn1->initializers[i1]);
	nn1->initializers[i1] = initializer_clone(nn2->initializers[i2]);
	nn1->in_types[i1] = nn2->in_types[i2];
}

void nn_clone_arg_o_from_o(nn_t nn1, uint o1, nn_t nn2, uint o2)
{
	if (NULL != nn1->out_names[o1])
		xfree(nn1->out_names[o1]);

	if (NULL != nn2->out_names[o2]) {

		PTR_ALLOC(char[strlen(nn2->out_names[o2]) + 1], name);
		strcpy(*name, nn2->out_names[o2]);
		nn1->out_names[o1] = *PTR_PASS(name);
	} else {

		nn1->out_names[o1] = NULL;
	}

	nn1->out_types[o1] = nn2->out_types[o2];
}

nn_t nn_clone(nn_t op)
{
	auto result = nn_from_nlop(op->network);

	for (uint i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (uint i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	return result;
}

void nn_clone_args(nn_t dst, nn_t src)
{
	auto result = dst;

	uint II = MIN(nn_get_nr_in_args(dst), nn_get_nr_in_args(src));
	uint OO = MIN(nn_get_nr_out_args(dst), nn_get_nr_out_args(src));

	for (uint i = 0; i < II; i++)
		nn_clone_arg_i_from_i(result, i, src, i);
	for (uint i = 0; i < OO; i++)
		nn_clone_arg_o_from_o(result, i, src, i);
}


static int get_index_from_name(int N, const char* names[N], const char* name)
{
	for (int i = 0; i < N; i++)
		if ((names[i] != NULL) && (names[i] != NULL) && ((0 == strcmp(name, names[i]))))
			return i;
	return -1;
}

unsigned int nn_get_nr_in_args(nn_t op)
{
	return nlop_get_nr_in_args(op->network);
}

unsigned int nn_get_nr_out_args(nn_t op)
{
	return nlop_get_nr_out_args(op->network);
}

unsigned int nn_get_nr_named_in_args(nn_t op)
{
	int result = 0;

	for (int i = 0; i < nlop_get_nr_in_args(op->network); i++)
		if (NULL != op->in_names[i])
			result++;
	return result;
}

unsigned int nn_get_nr_named_out_args(nn_t op)
{
	int result = 0;

	for (int i = 0; i < nlop_get_nr_out_args(op->network); i++)
		if (NULL != op->out_names[i])
			result++;
	return result;
}

unsigned int nn_get_nr_unnamed_in_args(nn_t op)
{
	return nn_get_nr_in_args(op) - nn_get_nr_named_in_args(op);
}

unsigned int nn_get_nr_unnamed_out_args(nn_t op)
{
	return nn_get_nr_out_args(op) - nn_get_nr_named_out_args(op);
}

bool nn_is_num_in_index(nn_t op, unsigned int i)
{
	return i < (unsigned int)nn_get_nr_unnamed_in_args(op);
}
bool nn_is_num_out_index(nn_t op, unsigned int o)
{
	return o < (unsigned int)nn_get_nr_unnamed_out_args(op);
}

int nn_get_out_arg_index(nn_t op, int o, const char* oname)
{
	if (NULL != oname) {

		assert((-1 == o) || (0 == o)); // index is ignored anyway
		o = get_index_from_name(nlop_get_nr_out_args(op->network), op->out_names, oname);
		if (-1 == o)
			error("Name %s not found!", oname);
	} else {
		assert(o >= -(int)nn_get_nr_unnamed_out_args(op));
		o = o + ((o < 0) ? (int)nn_get_nr_unnamed_out_args(op) : 0);

		assert(o < (int)nn_get_nr_unnamed_out_args(op));
		for (int i = 0; i <= o; i++)
			if (NULL != op->out_names[i])
				o++;
	}

	return o;
}

int nn_get_in_arg_index(nn_t op, int i, const char* iname)
{
	if (NULL != iname) {

		assert((-1 == i) || (0 == i)); // index is ignored anyway
		i = get_index_from_name(nlop_get_nr_in_args(op->network), op->in_names, iname);
		if (-1 == i)
			error("Name %s not found!", iname);
	} else {
		assert(i >= -(int)nn_get_nr_unnamed_in_args(op));
		i = i + ((i < 0) ? (int)nn_get_nr_unnamed_in_args(op) : 0);

		assert(i < (int)nn_get_nr_unnamed_in_args(op));
		for (int ii = 0; ii <= i; ii++)
			if (NULL != op->in_names[ii])
				i++;
	}

	return i;
}


const char* nn_get_in_name_from_arg_index(nn_t op, int i)
{
	assert(i < nlop_get_nr_in_args(op->network));
	return op->in_names[i];
}

const char* nn_get_out_name_from_arg_index(nn_t op, int o)
{
	assert(o < nlop_get_nr_out_args(op->network));
	return op->out_names[o];
}

int nn_get_in_index_from_arg_index(nn_t op, int i)
{
	if (NULL != op->in_names[i])
		return 0;

	int result = 0;

	while (i > 0)
		if (NULL == op->in_names[--i])
			result ++;

	return result;
}

int nn_get_out_index_from_arg_index(nn_t op, int o)
{
	if (NULL != op->out_names[o])
		return 0;

	int result = 0;

	while (o > 0)
		if (NULL == op->out_names[--o])
			result ++;

	return result;
}

static bool is_name_in_list(int N, const char* names[N], const char* name)
{
	bool result = false;
	for (int i = 0; i < N; i++)
		result |= (NULL == names[i]) ? false : (0 == strcmp(names[i], name));
	return result;
}

bool nn_is_name_in_in_args(nn_t op, const char* name)
{
	if (0 == nn_get_nr_named_in_args(op))
		return false;

	return is_name_in_list(nn_get_nr_in_args(op), op->in_names, name);
}

bool nn_is_name_in_out_args(nn_t op, const char* name)
{
	if (0 == nn_get_nr_named_out_args(op))
		return false;

	return is_name_in_list(nn_get_nr_out_args(op), op->out_names, name);
}


static int find_first_free_in_name_index(nn_t op, const char* prefix)
{
	int result = -1;
	bool valid = false;
	while (!valid) {

		result++;
		char tmp_name[strlen(prefix) + 10];
		sprintf(tmp_name, "%s%d", prefix, result);
		valid = !nn_is_name_in_in_args(op, tmp_name);
	}

	return result;
}

static int find_first_free_out_name_index(nn_t op, const char* prefix)
{
	int result = -1;
	bool valid = false;
	while (!valid) {

		result++;
		char tmp_name[strlen(prefix) + 10];
		sprintf(tmp_name, "%s%d", prefix, result);
		valid = !nn_is_name_in_out_args(op, tmp_name);
	}

	return result;
}


static nn_t nn_set_input_name(nn_t op, int i, const char* name)
{
	char tmp_name[strlen(name) + 10];
	if ('_' == name[strlen(name) - 1]) {

		int index = find_first_free_in_name_index(op, name);
		sprintf(tmp_name, "%s%d", name, index);
	} else {

		sprintf(tmp_name, "%s", name);
	}

	auto result = nn_clone(op);

	i = nn_get_in_arg_index(result, i, NULL);

	PTR_ALLOC(char[strlen(tmp_name) + 1], nname);
	strcpy(*nname, tmp_name);
	result->in_names[i] = *PTR_PASS(nname);

	return result;
}

static nn_t nn_set_output_name(nn_t op, int o, const char* name)
{
	char tmp_name[strlen(name) + 10];
	if ('_' == name[strlen(name) - 1]) {

		int index = find_first_free_out_name_index(op, name);
		sprintf(tmp_name, "%s%d", name, index);
	} else {

		sprintf(tmp_name, "%s", name);
	}

	auto result = nn_clone(op);

	o = nn_get_out_arg_index(result, o, NULL);

	PTR_ALLOC(char[strlen(tmp_name) + 1], nname);
	strcpy(*nname, tmp_name);
	result->out_names[o] = *PTR_PASS(nname);

	return result;
}

nn_t nn_set_input_name_F(nn_t op, int i, const char* name)
{
	auto result = nn_set_input_name(op, i, name);
	nn_free(op);
	return result;
}

nn_t nn_set_output_name_F(nn_t op, int o, const char* name)
{
	auto result = nn_set_output_name(op, o, name);
	nn_free(op);
	return result;
}

nn_t nn_unset_input_name_F(nn_t op, const char* name)
{
	int i = nn_get_in_arg_index(op, 0, name);
	auto result = nn_clone(op);

	xfree(result->in_names[i]);
	result->in_names[i] = NULL;
	result = nn_shift_input_index_F(op, nn_get_nr_in_args(op) - 1, i);

	nn_free(op);
	return result;
}

nn_t nn_unset_output_name_F(nn_t op, const char* name)
{
	int i = nn_get_out_arg_index(op, 0, name);
	auto result = nn_clone(op);

	xfree(result->out_names[i]);
	result->out_names[i] = NULL;
	result = nn_shift_output_index_F(op, nn_get_nr_out_args(op) - 1, i);

	nn_free(op);
	return result;
}

nn_t nn_rename_input_F(nn_t op, const char* nname, const char* oname)
{
	int i = nn_get_in_arg_index(op, 0, oname);

	auto result = nn_clone(op);

	xfree(result->in_names[i]);
	PTR_ALLOC(char[strlen(nname) + 1], nnname);
	strcpy(*nnname, nname);
	result->in_names[i] = *PTR_PASS(nnname);

	nn_free(op);

	return result;
}

nn_t nn_rename_output_F(nn_t op, const char* nname, const char* oname)
{
	int o = nn_get_out_arg_index(op, 0, oname);

	auto result = nn_clone(op);

	xfree(result->out_names[o]);
	PTR_ALLOC(char[strlen(nname) + 1], nnname);
	strcpy(*nnname, nname);
	result->out_names[o] = *PTR_PASS(nnname);

	nn_free(op);

	return result;
}

nn_t nn_set_initializer_F(nn_t op, int i, const char* iname, const struct initializer_s* ini)
{
	auto result = nn_clone(op);
	i = nn_get_in_arg_index(result, i, iname);
	result->initializers[i] = ini;
	nn_free(op);
	return result;
}

nn_t nn_set_in_type_F(nn_t op, int i, const char* iname, enum IN_TYPE in_type)
{
	auto result = nn_clone(op);
	i = nn_get_in_arg_index(result, i, iname);
	result->in_types[i] = in_type;
	nn_free(op);
	return result;
}

nn_t nn_set_out_type_F(nn_t op, int o, const char* oname, enum OUT_TYPE out_type)
{
	auto result = nn_clone(op);
	o = nn_get_out_arg_index(result, o, oname);
	result->out_types[o] = out_type;
	nn_free(op);
	return result;
}

const char** nn_get_out_names(nn_t op) {

	return op->out_names;
}

const char** nn_get_in_names(nn_t op) {

	return op->in_names;
}

const struct iovec_s* nn_generic_domain(nn_t op, int i, const char* iname)
{
	i = nn_get_in_arg_index(op, i, iname);
	return nlop_generic_domain(op->network, i);
}

const struct iovec_s* nn_generic_codomain(nn_t op, int o, const char* oname)
{
	o = nn_get_out_arg_index(op, o, oname);
	return nlop_generic_codomain(op->network, o);
}

void nn_debug(enum debug_levels dl, nn_t x)
{
	int II = nn_get_nr_in_args(x);

	debug_printf(dl, "NN\ninputs: %d\n", II);

	for (int i = 0, index = 0; i < II; i++) {

		auto io = nlop_generic_domain(x->network, i);
		char index_name[17];
		sprintf(index_name, "INDEX %d", index);
		debug_printf(dl, "%-15s", (NULL == x->in_names[i]) ? index_name : x->in_names[i]);
		debug_print_dims(dl, io->N, io->dims);

		if (NULL == x->in_names[i])
			index++;
	}

	int OO = nn_get_nr_out_args(x);

	debug_printf(dl, "outputs: %d\n", OO);

	for (int o = 0, index = 0; o < OO; o++) {

		auto io = nlop_generic_codomain(x->network, o);

		char index_name[17];
		sprintf(index_name, "INDEX %d", index);
		debug_printf(dl, "%-15s", (NULL == x->out_names[o]) ? index_name : x->out_names[o]);

		debug_print_dims(dl, io->N, io->dims);

		if (NULL == x->out_names[o])
			index++;
	}
}

int nn_get_nr_weights(nn_t op)
{
	int result = 0;
	for (uint i = 0; i < nn_get_nr_in_args(op); i++){

		if (NULL == op->initializers[i])
			assert((IN_OPTIMIZE != op->in_types[i]) && (IN_BATCHNORM != op->in_types[i]));
		else
			result++;
	}
	return result;
}

void nn_get_in_types(nn_t op, uint N, enum IN_TYPE in_types[N])
{
	assert(N == nn_get_nr_in_args(op));

	for (uint i = 0; i < N; i++)
		in_types[i] = op->in_types[i];
}

void nn_get_out_types(nn_t op, uint N, enum OUT_TYPE out_types[N])
{
	assert(N == nn_get_nr_out_args(op));

	for (uint i = 0; i < N; i++)
		out_types[i] = op->out_types[i];
}

nn_t nn_checkpoint_F(nn_t op, bool der_once)
{
	auto result = nn_from_nlop(nlop_checkpoint_create(op->network, der_once));;

	for (uint i = 0; i < nn_get_nr_in_args(result); i++)
		nn_clone_arg_i_from_i(result, i, op, i);
	for (uint i = 0; i < nn_get_nr_out_args(result); i++)
		nn_clone_arg_o_from_o(result, i, op, i);

	nn_free(op);

	return result;
}

void nn_export_graph(const char* filename, nn_t op, graph_t opts)
{
	int II = nlop_get_nr_in_args(op->network);
	int OO = nlop_get_nr_out_args(op->network);

	unsigned int D[II + OO];
	const char** arg_nodes[II + OO];

	const char* str = operator_get_graph_string(op->network->op, II + OO, D, arg_nodes, opts);

	FILE *fp;
	fp = fopen(filename, "w+");

	assert(0 != fp);

	fprintf(fp, "digraph { \n");

	fprintf(fp, "{\n%s}\n", str);

	int counter_input = 0;
	int counter_weight = 0;


	for (int o = 0; o < OO; o++) {

		fprintf(fp, "%s -> Output_%d;\n", (arg_nodes[o])[0], o);
		xfree((arg_nodes[o])[0]);
		xfree((arg_nodes[o]));
		assert(1 == D[o]);
	}

	for (int i = 0; i < II; i++) {

		if ((IN_OPTIMIZE == op->in_types[i]) || (IN_BATCHNORM == op->in_types[i])) {

			for (int j = 0; j < (int)D[OO + i]; j++) {

				fprintf(fp, "Weight_%d -> %s;\n", counter_weight, (arg_nodes[OO + i])[j]);	
				xfree((arg_nodes[OO + i])[j]);
			}
			counter_weight++;
		} else {

			for (int j = 0; j < (int)D[OO + i]; j++) {

				fprintf(fp, "Input_%d -> %s;\n", counter_input, (arg_nodes[OO + i])[j]);	
				xfree((arg_nodes[OO + i])[j]);
			}
			counter_input++;
		}
		xfree((arg_nodes[OO + i]));
	}

	int index = 0;
	if (0 < counter_input) {

		fprintf(fp, "subgraph cluster_inputs{ label = \"Inputs\";\n rank = source;\n");
		for (int i = 0; i < counter_input; i++, index ++) {

			while ((IN_OPTIMIZE == op->in_types[index]) || (IN_BATCHNORM == op->in_types[index]))
				index++;
			if (NULL != op->in_names[index])
				fprintf(fp, "Input_%d [shape = diamond, label = \"%s\"];\n", i, op->in_names[index]);
			else
				fprintf(fp, "Input_%d [shape = diamond];\n", i);
		}

		fprintf(fp, "\n}\n");
	}

	if (0 < OO) {

		fprintf(fp, "subgraph cluster_outputs{ label = \"Outputs\";\n rank = sink;\n");
		for (int i = 0; i < OO; i++) {

			if (NULL != op->out_names[i])
				fprintf(fp, "Output_%d [shape = diamond, label = \"%s\"];\n", i, op->out_names[i]);
			else
				fprintf(fp, "Output_%d [shape = diamond];\n", i);
		}
		fprintf(fp, "edge[ style=invis];\nOutput_0");
		for (int i = 1; i < OO; i++)
				fprintf(fp, " -> Output_%d", i);
		fprintf(fp, "\n}\n");
	}



	index = 0;
	if (0 < counter_weight) {

		fprintf(fp, "subgraph cluster_weights{\n label = \"Weights\";\n rank = source;\n");

		for (int i = 0; i < counter_weight; i++, index ++) {

			while (!((IN_OPTIMIZE == op->in_types[index]) || (IN_BATCHNORM == op->in_types[index])))
				index++;
			if (NULL != op->in_names[index])
				fprintf(fp, "Weight_%d [shape = diamond, label = \"%s\"];\n", i, op->in_names[index]);
			else
				fprintf(fp, "Weight_%d [shape = diamond];\n", i);
		}
		fprintf(fp, "edge[ style=invis];\nWeight_0");
		for (int i = 1; i < counter_weight; i++)
				fprintf(fp, " -> Weight_%d", i);
		fprintf(fp, "\n}\n");
	}
	fprintf(fp, " }");

	fclose(fp);
}