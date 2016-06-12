
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "misc/misc.h"
#include "misc/mmio.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "flags <arg1> <arg2>";
static const char help_str[] = "Estimate sub-pixel shift.";


int main_estshift(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 3, usage_str, help_str);

	unsigned int flags = atoi(argv[1]);

	long dims1[DIMS];
	long dims2[DIMS];

	const complex float* in1 = load_cfl(argv[2], DIMS, dims1);
	const complex float* in2 = load_cfl(argv[3], DIMS, dims2);

	assert(md_check_compat(DIMS, ~0u, dims1, dims2));

	complex float* tmp1 = md_alloc(DIMS, dims1, CFL_SIZE);
	complex float* tmp2 = md_alloc(DIMS, dims2, CFL_SIZE);

	fftuc(DIMS, dims1, flags, tmp1, in1);
	fftuc(DIMS, dims2, flags, tmp2, in2);

	md_zmulc(DIMS, dims1, tmp1, tmp1, tmp2);

	printf("Shifts:");

	for (unsigned int i = 0; i < DIMS; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		long shift[DIMS] = { 0 };
		shift[i] = 1;

		md_circ_shift(DIMS, dims1, shift, tmp2, tmp1, CFL_SIZE);

		// the weighting is not optimal due to double squaring
		// and we compute finite differences (filter?)

		complex float sc = md_zscalar(DIMS, dims1, tmp2, tmp1);
		float sh = cargf(sc) / (2. * M_PI) * (float)dims1[i];

		printf("\t%f", sh);
	}

	printf("\n");

	md_free(tmp1);
	md_free(tmp2);

	unmap_cfl(DIMS, dims1, in1);
	unmap_cfl(DIMS, dims2, in2);

	exit(0);
}


