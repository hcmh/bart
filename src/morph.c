
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/init.h"

#include "misc/mri.h"

#include "nlops/nlop.h"
#include "nlops/conv.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

static void mask_conv(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{

	const struct nlop_s* nlop_conv = nlop_convcorr_geom_create(D, (READ_FLAG|PHS1_FLAG|PHS2_FLAG), dims, dims, mask_dims,
								PAD_SAME, false, NULL, NULL, 'N');

	nlop_generic_apply_unchecked(nlop_conv, 3, (void*[3]){out, (void*)in, mask});

	nlop_free(nlop_conv);
}

static void erosion2(unsigned int D, const long dims[D], float level, complex float* out, const complex float* in)
{
	long size = md_calc_size(DIMS, dims) * 2;

	const float* inf = (const float*)in;
	float* outf = (float*)out;

	#pragma omp parallel for
		for (long i = 0; i < size; i++)
			outf[i] = inf[i] == level ? 1. : 0.;
}

static void erosion(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{
	complex float* tmp_data = md_alloc(DIMS, dims, CFL_SIZE);

	mask_conv(D, mask_dims, mask, dims, tmp_data, in);

	erosion2(D, dims,  md_zasum(D, mask_dims, mask), out, tmp_data);

	md_free(tmp_data);
}

static void dilation2(unsigned int D, const long dims[D], complex float* out, const complex float* in)
{
	long size = md_calc_size(DIMS, dims) * 2;

	const float* inf = (const float*)in;
	float* outf = (float*)out;

	#pragma omp parallel for
		for (long i = 0; i < size; i++)
			outf[i] = inf[i] >= 1 ? 1. : 0.;
}

static void dilation(unsigned int D, const long mask_dims[D], complex float* mask, const long dims[D], complex float* out, const complex float* in)
{
	complex float* tmp_data = md_alloc(DIMS, dims, CFL_SIZE);

	mask_conv(D, mask_dims, mask, dims, tmp_data, in);

	dilation2(D, dims, out, tmp_data);

	md_free(tmp_data);
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
	enum morph_type { NONE, EROSION, DILATION, OPENING, CLOSING } morph_type = NONE;

	enum mask_type { HLINE, VLINE, CROSS, BLOCK } mask_type = BLOCK;


	const struct opt_s opts[] = {

		OPT_SELECT('e', enum morph_type, &morph_type, EROSION, "EROSION"),
		OPT_SELECT('d', enum morph_type, &morph_type, DILATION, "DILATION"),
		OPT_SELECT('o', enum morph_type, &morph_type, OPENING, "OPENING"),
		OPT_SELECT('c', enum morph_type, &morph_type, CLOSING, "CLOSING"),
		OPT_SELECT('b', enum mask_type, &mask_type, BLOCK, "mask type: BLOCK (default"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int N = DIMS;

	long dims[N];

	complex float* in = load_cfl(argv[2], N, dims);

	complex float* out = create_cfl(argv[3], N, dims);

	int mask_size = atoi(argv[1]);

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


