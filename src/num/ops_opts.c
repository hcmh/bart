#include <stdbool.h>
#include <stdio.h>

#include "num/multind.h"

#include "ops_opts.h"





void print_operator_run_flags(int N, operator_run_opt_flags_t run_opts[N][N])
{
	printf("%d x %d runopts = \n", N, N);
	for (int i = 0; i < N; i++) {

		printf("[");

		for (int j = 0; j < N; j++)
			printf("%lu ", (unsigned long)run_opts[i][j]);

		printf("]\n");
	}
}
void print_operator_io_run_flags(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI])
{
	operator_run_opt_flags_t io_run_opts[NO][NI];
	operator_run_opts_to_io_run_opts(NO, NI, io_flags, io_run_opts, run_opts);

	operator_io_run_opts_to_run_opts(NO, NI, io_flags, run_opts, io_run_opts);

	operator_run_opts_to_io_run_opts(NO, NI, io_flags, io_run_opts, run_opts);


	printf("NO=%d x NI=%d io runopts = \n", NO, NI);
	for (int o = 0; o < NO; o++) {

		printf("[");

		for (int i = 0; i < NI; i++)
			printf("%lu ", (unsigned long)io_run_opts[o][i]);

		printf("]\n");
	}
}


void operator_run_opts_to_io_run_opts(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t io_run_opts[NO][NI], operator_run_opt_flags_t run_opts[NO + NI][NO + NI])
{

	int in_index = 0;
	for (int i = 0; i < NO + NI; i++) {

		int out_index = 0;

		if (MD_IS_SET(io_flags, i))
			continue;

		for (int j = 0; j < NO + NI; j++) {

			if (!MD_IS_SET(io_flags, j))
				continue;
			
			//printf("NO = %d, NI= %d, out_index = %d, in_index = %d, i = %d, j = %d\n", NO, NI, out_index, in_index, i, j);

			io_run_opts[out_index][in_index] = run_opts[i][j] & run_opts[j][i];

			out_index ++;
		}

		in_index++;
	}
}

void operator_io_run_opts_to_run_opts(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], operator_run_opt_flags_t io_run_opts[NO][NI])
{
	
	int in_index = 0;
	for (int i = 0; i < NO + NI; i++) {

		if (MD_IS_SET(io_flags, i))
			continue;

		int out_index = 0;
		for (int j = 0; j < NO + NI; j++) {

			if (!MD_IS_SET(io_flags, j))
				continue;

			run_opts[i][j] = io_run_opts[out_index][in_index];
			run_opts[j][i] = io_run_opts[out_index][in_index];

			out_index ++;
		}
		in_index++;
	}
}


unsigned int operator_index_to_io_index(unsigned int io_flags, unsigned int index, bool output)
{
	assert(output == (bool)MD_IS_SET(io_flags, index));
	unsigned int io_index = 0;
	
	for (unsigned int i = 0; i < index; i++)
		if (output == (bool)MD_IS_SET(io_flags, i))
			io_index++;
	
	return io_index;
}

unsigned int operator_io_index_to_index(unsigned int io_flags, unsigned int io_index, bool output)
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

void operator_init_run_opts(int N, operator_run_opt_flags_t run_opts[N][N])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			run_opts[i][j] = 0;
}

void operator_set_run_opt(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	run_opts[i][j] = MD_SET(run_opts[i][j], option);
}

void operator_set_run_opt_sym(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	run_opts[i][j] = MD_SET(run_opts[i][j], option);
	run_opts[j][i] = MD_SET(run_opts[j][i], option);
}

void operator_set_oi_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int oind = operator_io_index_to_index(io_flags, o, true);
	unsigned int iind = operator_io_index_to_index(io_flags, i, false);
	
	operator_set_run_opt_sym(NO + NI, run_opts, oind, iind, option);
}

void operator_set_oo_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int oind1 = operator_io_index_to_index(io_flags, o1, true);
	unsigned int oind2 = operator_io_index_to_index(io_flags, o2, true);
	
	operator_set_run_opt(NO + NI, run_opts, oind1, oind2, option);
}

void operator_set_ii_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int iind1 = operator_io_index_to_index(io_flags, i1, false);
	unsigned int iind2 = operator_io_index_to_index(io_flags, i2, false);
	
	operator_set_run_opt(NO + NI, run_opts, iind1, iind2, option);
}

void operator_unset_run_opt(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	run_opts[i][j] = MD_CLEAR(run_opts[i][j], option);
}

void operator_unset_run_opt_sym(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	run_opts[i][j] = MD_CLEAR(run_opts[i][j], option);
	run_opts[j][i] = MD_CLEAR(run_opts[j][i], option);
}

void operator_unset_oi_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int oind = operator_io_index_to_index(io_flags, o, true);
	unsigned int iind = operator_io_index_to_index(io_flags, i, false);
	
	operator_unset_run_opt_sym(NO + NI, run_opts, oind, iind, option);
}

void operator_unset_oo_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int oind1 = operator_io_index_to_index(io_flags, o1, true);
	unsigned int oind2 = operator_io_index_to_index(io_flags, o2, true);
	
	operator_unset_run_opt(NO + NI, run_opts, oind1, oind2, option);
}

void operator_unset_ii_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int iind1 = operator_io_index_to_index(io_flags, i1, false);
	unsigned int iind2 = operator_io_index_to_index(io_flags, i2, false);
	
	operator_unset_run_opt(NO + NI, run_opts, iind1, iind2, option);
}

bool operator_get_run_opt(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	return MD_IS_SET(run_opts[i][j], option);
}

bool operator_get_run_opt_sym(int N, operator_run_opt_flags_t run_opts[N][N], unsigned int i, unsigned int j, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	return MD_IS_SET(run_opts[i][j], option) && MD_IS_SET(run_opts[j][i], option); 
}

bool operator_get_oi_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o, unsigned int i, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int oind = operator_io_index_to_index(io_flags, o, true);
	unsigned int iind = operator_io_index_to_index(io_flags, i, false);

	return operator_get_run_opt_sym(NO + NI, run_opts, oind, iind, option);
}
bool operator_get_oo_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int o1, unsigned int o2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int oind1 = operator_io_index_to_index(io_flags, o1, true);
	unsigned int oind2 = operator_io_index_to_index(io_flags, o2, true);

	return operator_get_run_opt(NO + NI, run_opts, oind1, oind2, option);
}

bool operator_get_ii_run_opt(int NO, int NI, unsigned int io_flags, operator_run_opt_flags_t run_opts[NO + NI][NO + NI], unsigned int i1, unsigned int i2, enum OPERATOR_RUN_OPT_FLAGS_INDEX option)
{
	unsigned int iind1 = operator_io_index_to_index(io_flags, i1, false);
	unsigned int iind2 = operator_io_index_to_index(io_flags, i2, false);

	return operator_get_run_opt(NO + NI, run_opts, iind1, iind2, option);
}

