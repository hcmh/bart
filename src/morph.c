
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/init.h"
#include "num/conv.h"

#include "misc/mri.h"

#include "nlops/nlop.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

static void mask_conv(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{
	struct conv_plan* plan = conv_plan(D, FFT_FLAGS, CONV_CYCLIC, CONV_SYMMETRIC, dims, dims, mask_dims, mask);

	conv_exec(plan, out, in);

	conv_free(plan);
}

static void erosion(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{
	complex float* tmp = md_alloc(DIMS, dims, CFL_SIZE);

	mask_conv(D, mask_dims, mask, dims, tmp, in);

	// take relative error into account due to floating points
	md_zsgreatequal(D, dims, out, tmp, (1 - 0.00001) * md_zasum(D, mask_dims, mask));

	md_free(tmp);
}


static void dilation(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{
	complex float* tmp = md_alloc(DIMS, dims, CFL_SIZE);

	mask_conv(D, mask_dims, mask, dims, tmp, in);

	// take relative error into account due to floating points
	md_zsgreatequal(D, dims, out, tmp, (1 - md_zasum(D, mask_dims, mask) * 0.00001));

	md_free(tmp);
}

static void opening(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{
	complex float* tmp_data = md_alloc(DIMS, dims, CFL_SIZE);

	erosion(D, mask_dims, mask, dims, tmp_data, in);

	dilation(D, mask_dims, mask, dims, out, tmp_data);

	md_free(tmp_data);
}

static void closing(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{
	complex float* tmp_data = md_alloc(DIMS, dims, CFL_SIZE);

	dilation(D, mask_dims, mask, dims, tmp_data, in);

	erosion(D, mask_dims, mask, dims, out, tmp_data);

	md_free(tmp_data);
}

static const char usage_str[] = "mask_size <binary input> <binary output>";
static const char help_str[] = "Perform morphological operators on binary data.";



int main_morph(int argc, char* argv[])
{
	enum morph_type { EROSION, DILATION, OPENING, CLOSING } morph_type = EROSION;

	enum mask_type { HLINE, VLINE, CROSS, BLOCK } mask_type = BLOCK;


	const struct opt_s opts[] = {

		OPT_SELECT('e', enum morph_type, &morph_type, EROSION, "EROSION (default)"),
		OPT_SELECT('d', enum morph_type, &morph_type, DILATION, "DILATION"),
		OPT_SELECT('o', enum morph_type, &morph_type, OPENING, "OPENING"),
		OPT_SELECT('c', enum morph_type, &morph_type, CLOSING, "CLOSING"),
		OPT_SELECT('b', enum mask_type, &mask_type, BLOCK, "mask type: BLOCK (default)"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int N = DIMS;

	long dims[N];

	complex float* in = load_cfl(argv[2], N, dims);

	complex float* out = create_cfl(argv[3], N, dims);

	int mask_size = atoi(argv[1]);

	// Only odd mask size values are supportet for the convolution
	assert(1 == mask_size%2);

	// FIXME: Check if data is binary else Raise
	// ...

	long mask_dims[N];
	md_set_dims(N, mask_dims, 1);
	mask_dims[READ_DIM] = mask_size;
	mask_dims[PHS1_DIM] = mask_size;

	complex float* mask = md_alloc(DIMS, mask_dims, CFL_SIZE);
	md_clear(N, mask_dims, mask, CFL_SIZE);

	switch (mask_type) {

		case HLINE:
			printf("Mask Type is not implemented yet.\n");
			// mask = {{0, 0, 0},
			// 	{1, 1, 1},
			// 	{0, 0, 0}};
			break;

		case VLINE:
			printf("Mask Type is not implemented yet.\n");
			// mask = {{0, 1, 0},
			// 	{0, 1, 0},
			// 	{0, 1, 0}};
			break;

		case CROSS:
			printf("Mask Type is not implemented yet.\n");
			// mask = {{0, 1, 0},
			// 	{1, 1, 1},
			// 	{0, 1, 0}};
			break;

		case BLOCK:
			md_zfill(N, mask_dims, mask, 1.);
			break;

		default:
			printf("Please choose a correct structural element/mask.\n");
			break;
	}

	switch (morph_type) {

		case EROSION:
			erosion(N, mask_dims, mask, dims, out, in);
			break;

		case DILATION:
			dilation(N, mask_dims, mask, dims, out, in);
			break;

		case OPENING:
			opening(N, mask_dims, mask, dims, out, in);
			break;

		case CLOSING:
			closing(N, mask_dims, mask, dims, out, in);
			break;

		default:
			printf("Please choose a morphological operation.\n");
			break;
	}

	md_free(mask);

	unmap_cfl(N, dims, in);
	unmap_cfl(N, dims, out);
	return 0;
}


