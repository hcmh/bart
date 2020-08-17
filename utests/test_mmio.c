#include <stdio.h>

#include "complex.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "num/multind.h"
#include "num/rand.h"
#include "num/flpmath.h"

#include "utest.h"


static int create_multi(void)
{
	char filename[] = "utest_mmio_file_tmp_apofghpad9hg9s";
	char filename_cfl[] = "utest_mmio_file_tmp_apofghpad9hg9s.cfl";
	char filename_hdr[] = "utest_mmio_file_tmp_apofghpad9hg9s.hdr";

	long dims1[3] = { 2, 1, 5};
	long dims2[2] = { 3, 2};

	complex float* ptr1 = md_alloc(3, dims1, sizeof(complex float));
	complex float* ptr2 = md_alloc(2, dims2, sizeof(complex float));

	md_zfill(3, dims1, ptr1, 1.);
	md_gaussian_rand(2, dims2, ptr2);

	dump_multi_cfl(filename, 2, MAKE_ARRAY(3u, 2u), MAKE_ARRAY((const long*)dims1, (const long*)dims2), MAKE_ARRAY((const complex float*)ptr1, (const complex float*)ptr2));

	unsigned int N_max = 10;
	unsigned int D_max = 20;

	unsigned int D[N_max];
	long dims_loaded[N_max][D_max];
	complex float* args_loaded[N_max];

	unsigned int N = load_multi_cfl(filename, N_max, D_max, D, dims_loaded, args_loaded);

	bool ok = true;
	ok &= (2 == N);
	ok &= (3 == D[0]);
	ok &= (2 == D[1]);
	for (unsigned int i = 2; i < N_max; i++)
		ok &= (0 == D[i]);
	ok &= md_check_equal_dims(D[0], dims1, dims_loaded[0], ~0u);
	ok &= md_check_equal_dims(D[1], dims2, dims_loaded[1], ~0u);

	ok &= 1.e-8 > md_zrmse(D[0], dims_loaded[0], ptr1, args_loaded[0]);
	ok &= 1.e-8 > md_zrmse(D[1], dims_loaded[1], ptr2, args_loaded[1]);

	const long* dims_unmap[N];
	for (unsigned int i = 0; i < N; i++)
		dims_unmap[i] = &dims_loaded[i][0];
	unmap_multi_cfl(N, D, dims_unmap, args_loaded);

	long size;
	complex float* args2 = load_cfl(filename, 1, &size);

	ok &= 1.e-8 > md_zrmse(D[0], dims_loaded[0], ptr1, args2);
	ok &= 1.e-8 > md_zrmse(D[1], dims_loaded[1], ptr2, args2 + md_calc_size(D[0], dims_loaded[0]));

	unmap_cfl(1, &size, args2);
	ok &= (0 == remove(filename_cfl));
	ok &= (0 == remove(filename_hdr));
	debug_printf(DP_INFO, "Warning for overwriting is expected!\n");

	return ok;
}

UT_REGISTER_TEST(create_multi);
