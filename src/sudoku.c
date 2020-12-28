/* Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/rand.h"

#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/iter.h"
#include "iter/iter2.h"

#include "linops/someops.h"
#include "linops/sum.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/types.h"


static const char usage_str[] = "<board> <solution>";
static const char help_str[] = "Experimental optimization-based sudoku solver (may produce incorrect solutions).\n";




static void solution(complex float out[9][9], const complex float x[9][9][9])
{
	for (int i = 0; i < 9; i++) {

		for (int j = 0; j < 9; j++) {

			float max = 0.;
			int imax = 0;

			for (int k = 0; k < 9; k++) {

				float v = crealf(x[i][j][k]);

				if (v > max) {

					max = v;
					imax = k;
				}
			}

			out[i][j] = imax + 1;;
		}
	}
}

static void print_board(const complex float b[9][9])
{
	for (int i = 0; i < 9; i++) {

		if (0 == i % 3)
			bart_printf("+-------+-------+-------+\n");

		for (int j = 0; j < 9; j++) {
			
			if (0 == j % 3)
				bart_printf("| ");

			bart_printf("%d ", (int)crealf(b[i][j]));
		}

		bart_printf("|\n");
	}

	bart_printf("+-------+-------+-------+\n");
}

int main_sudoku(int argc, char* argv[argc])
{
	float lambda = 0.001;
	float rho = 0.01;
	int iter = 100;

	const struct opt_s opts[] = {

		OPT_FLOAT('l', &lambda, "l", "lambda"),
		OPT_FLOAT('r', &rho, "r", "rho"),		
		OPT_INT('i', &iter, "i", "iter"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long idims[2];
	complex float* in = load_cfl(argv[1], 2, idims);

	//print_board(MD_CAST_ARRAY2(complex float, 2, odims, in, 0, 1));

	assert((9 == idims[0]) && (9 == idims[1]));

	complex float sudoku[9][9][9] = { 0 };	// lifted sudoku state
	complex float pattern[9][9] = { 0 };
	complex float cn[9][9];

	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {

			complex float ival = in[i * 9 + j];

			assert(0. == cimagf(ival));
			assert(0. <= crealf(ival));
			assert(9. >= crealf(ival));

			int ind = crealf(ival);

			assert(ind == crealf(ival));

			cn[i][j] = 1. / 9.;

			if (1 < ind) {

				pattern[i][j] = 1.;
				sudoku[i][j][ind - 1] = 1.;
			}
		}
	}

	unmap_cfl(2, idims, in);


	long dims[5] = { 9, 3, 3, 3, 3 };


	const struct operator_p_s* prox[] = {

		prox_lineq_create(linop_cdiag_create(5, dims, 30u, &pattern[0][0]), &sudoku[0][0][0]),
		prox_lineq_create(linop_avg_create(5, dims, MD_BIT(3)|MD_BIT(4)), &cn[0][0]),
		prox_lineq_create(linop_avg_create(5, dims, MD_BIT(1)|MD_BIT(2)), &cn[0][0]),
		prox_lineq_create(linop_avg_create(5, dims, MD_BIT(1)|MD_BIT(3)), &cn[0][0]),
		prox_lineq_create(linop_avg_create(5, dims, MD_BIT(0)), &cn[0][0]),
		prox_nonneg_create(5, dims),
		prox_thresh_create(5, dims, lambda, 0LU),
	};

	const struct linop_s* eye = linop_identity_create(5, dims);

	const struct linop_s* ops[ARRAY_SIZE(prox)];

	for (unsigned int i = 0; i < ARRAY_SIZE(prox); i++)
		ops[i] = eye;


	long bdims[3] = { 9, 9, 9 };
	complex float* x = md_calloc(3, bdims, CFL_SIZE);
//	md_uniform_rand(3, bdims, x);

	struct iter_admm_conf conf = iter_admm_defaults;
	conf.rho = rho;
	conf.maxiter = iter;

	iter2_admm( CAST_UP(&conf), NULL, ARRAY_SIZE(prox), prox, ops, NULL, NULL,
		    md_calc_size(5, dims) * 2, (float*)x, NULL,  NULL);


	long odims[2] = { 9, 9 };
	complex float* out = create_cfl(argv[2], 2, odims);

	solution(MD_CAST_ARRAY2(complex float, 2, odims, out, 0, 1), 
		 MD_CAST_ARRAY3(complex float, 3, bdims, x, 0, 1, 2));

	md_free(x);

	print_board(MD_CAST_ARRAY2(complex float, 2, odims, out, 0, 1));

	return 0;
}


