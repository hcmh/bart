#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>

#include "misc/debug.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/io.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"



#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Combine/Split multiple cfl files to one multi-cfl file\n"
				"Either <input1> [... <inputN>] <output> or -s <input> <output1> [... <outputN>]";


int main_multicfl(int argc, char* argv[argc])
{
	long count = 0;
	const char** inout_files = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 1, OPT_STRING, sizeof(char*), &inout_files, "inout"),
	};

	bool separate = false;

	const struct opt_s opts[] = {

		OPT_SET('s', &separate, "separate"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (2 > count)
		error("Specify at least two files!\n");

	long num_single_files = count - 1;

	if (!separate) {

		int D[num_single_files];
		long dims_load[num_single_files][DIMS];
		const long* dims_store[num_single_files];
		const complex float* args[num_single_files];

		for (int i = 0; i < num_single_files; i++) {

			io_reserve_input(inout_files[i]);
			D[i] = DIMS;
			args[i] = load_cfl(inout_files[i], D[i], dims_load[i]);
			dims_store[i] = dims_load[i];
		}


		io_reserve_output(inout_files[num_single_files]);
		dump_multi_cfl(inout_files[num_single_files], num_single_files, D, dims_store, args);

		for (int i = 0; i < num_single_files; i++)
			unmap_cfl(D[i], dims_load[i], args[i]);

	} else {

		int D_max = DIMS;
		int D[num_single_files];
		long dims_load[num_single_files][D_max];
		const long* dims_store[num_single_files];
		complex float* args[num_single_files];

		io_reserve_input(inout_files[0]);
		int N = load_multi_cfl(inout_files[0], num_single_files, D_max, D, dims_load, args);
		if(N != num_single_files)
			error("Number of cfls in input does not match no of outputs!");

		for (int i = 0; i < num_single_files; i++) {

			const char* name = inout_files[i + 1];
			io_reserve_output(name);
			dump_cfl(name, D[i], dims_load[i], args[i]);
			dims_store[i] = dims_load[i];
		}

		unmap_multi_cfl(N, D, dims_store, args);
	}

	xfree(inout_files);

	return 0;
}


