/* Copyright 2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <sys/types.h>
#include <stdint.h>
#include <complex.h>
#include <fcntl.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/filter.h"

#include "linops/linop.h"

#include "noncart/nufft.h"

#include "calib/cc.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static void xread(int fd, void* buf, size_t size)
{
	if (size != (size_t)read(fd, buf, size))
		error("reading file");
}

static void xseek(int fd, off_t pos)
{
        if (-1 == lseek(fd, pos, SEEK_SET))
		error("seeking");
}

static void meas_setup(int fd)
{
	uint32_t start = 0;

	xseek(fd, 0);
	xread(fd, &start, sizeof(uint32_t));
	xseek(fd, start);
}


static int adc_read(int fd, const long dims[DIMS], long pos[DIMS], complex float* buf)
{
	uint32_t lc[8];
	xread(fd, lc, sizeof(lc));

	pos[PHS1_DIM]	= lc[0];
	pos[TE_DIM]	= lc[1];
	pos[SLICE_DIM]	= lc[2];
	pos[TIME_DIM]	= lc[3];
	pos[COEFF_DIM]	= lc[4];
	pos[TIME2_DIM]	= lc[5];

	pos[COIL_DIM] = 0;

	debug_print_dims(DP_DEBUG4, DIMS, pos);

	xread(fd, buf, dims[COIL_DIM] * dims[READ_DIM] * CFL_SIZE);

	return 0;
}




static const char usage_str[] = "<dat file> <output>";
//	fprintf(fd, "Usage: %s [...] [-a A] <dat file> <output>\n", name);

static const char help_str[] = "Read streaming data and reconstruct frame by frame.";


int main_rtreco(int argc, char* argv[argc])
{
	bool gpu = false;
	long turns = 5;

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);

	long V = -1;
	unsigned int median = 1;
	bool geom = true;

	bool mcoil = false;

	struct opt_s opts[] = {

		OPT_LONG('x', &(dims[READ_DIM]), "X", "number of samples (read-out)"),
		OPT_LONG('r', &(dims[PHS1_DIM]), "R", "numer of radial spokes / frame"),
		OPT_LONG('t', &turns, "T", "numer of turns"),
		OPT_UINT('m', &median, "L", "Median filter"),
		OPT_SET('C', &geom, "complex median"),
		OPT_SET('U', &mcoil, "uncombined output"),
		OPT_LONG('c', &(dims[COIL_DIM]), "C", "number of channels"),
		OPT_LONG('v', &V, "V", "number of virtual channels"),
		OPT_LONG('n', &(dims[6]), "N", "number of repetitions"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	debug_print_dims(DP_DEBUG1, DIMS, dims);

	if (-1 == V)
		V = dims[COIL_DIM];


        int ifd;
        if (-1 == (ifd = open(argv[1], O_RDONLY)))
                error("error opening file.");

	meas_setup(ifd);



	// Read trajectory
	long traj_dims[DIMS];
	complex float* traj = load_cfl("t", DIMS, traj_dims);

	assert(3 == traj_dims[0]);
	assert(0 == traj_dims[2] % turns);


	(gpu ? num_init_gpu : num_init)();


	long ksp_dims[DIMS];
	md_copy_dims(DIMS, ksp_dims, dims);

	assert(1 == ksp_dims[2]);
	ksp_dims[2] = ksp_dims[1];
	ksp_dims[1] = ksp_dims[0];
	ksp_dims[0] = 1;

	assert(md_check_compat(DIMS, ~(PHS1_FLAG|PHS2_FLAG), ksp_dims, traj_dims));



	// output and imput

	long coilim_dims[DIMS] = { [0 ... DIMS - 1]  = 1 };
	estimate_im_dims(DIMS, FFT_FLAGS, coilim_dims, traj_dims, traj);

	debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", coilim_dims[0], coilim_dims[1], coilim_dims[2]);
	coilim_dims[COIL_DIM] = V;


	long out_dims[DIMS] = { [0 ... DIMS - 1]  = 1 };
	md_copy_dims(3, out_dims, coilim_dims);
	md_copy_dims(DIMS - 4, out_dims + 4, dims + 4);

	out_dims[COIL_DIM] = mcoil ? V : 1;

	long img_dims[DIMS];
	md_select_dims(DIMS, ~MD_BIT(6), img_dims, out_dims);

	complex float* out = create_cfl(argv[2], DIMS, out_dims);

	complex float* cimg = md_alloc(DIMS, coilim_dims, CFL_SIZE);
	complex float* img = md_alloc(DIMS, img_dims, CFL_SIZE);



	long adc_dims[DIMS];
	md_select_dims(DIMS, PHS1_FLAG|COIL_FLAG, adc_dims, ksp_dims);

	void* adc = md_alloc(DIMS, adc_dims, CFL_SIZE);

	long buf_dims[DIMS];
	md_select_dims(DIMS, PHS1_FLAG|PHS2_FLAG|COIL_FLAG, buf_dims, ksp_dims);

	complex float* buf = md_calloc(DIMS, buf_dims, CFL_SIZE);

	// coil compression

	int channels = ksp_dims[COIL_DIM];

	long cc_dims[DIMS] = MD_INIT_ARRAY(DIMS, 1);

	cc_dims[COIL_DIM] = channels;
	cc_dims[MAPS_DIM] = channels;

	complex float* cc = md_alloc(DIMS, cc_dims, CFL_SIZE);

	long cc2_dims[DIMS];
	md_copy_dims(DIMS, cc2_dims, cc_dims);
	cc2_dims[MAPS_DIM] = V;


	debug_printf(DP_DEBUG1, "Compressing to %ld virtual coils...\n", V);

	long buf2_dims[DIMS];
	md_copy_dims(DIMS, buf2_dims, buf_dims);
	buf2_dims[COIL_DIM] = V;

	long bufT_dims[DIMS];
	md_copy_dims(DIMS, bufT_dims, buf_dims);
	bufT_dims[MAPS_DIM] = V;
	bufT_dims[COIL_DIM] = 1;

	complex float* buf2 = md_alloc(DIMS, buf_dims, CFL_SIZE);


	struct nufft_conf_s conf = nufft_conf_defaults;
	const struct linop_s* nufft_op = nufft_create(DIMS, buf2_dims, coilim_dims, traj_dims, traj, NULL, conf);

	// filter


	long filt_dims[DIMS];
	md_select_dims(DIMS, ~READ_FLAG, filt_dims, traj_dims);

	complex float* filter = md_alloc(DIMS, filt_dims, CFL_SIZE);
	md_zrss(DIMS, traj_dims, READ_FLAG, filter, traj);

	long buf2_strs[DIMS];
	md_calc_strides(DIMS, buf2_strs, buf2_dims, CFL_SIZE);

	long filt_strs[DIMS];
	md_calc_strides(DIMS, filt_strs, filt_dims, CFL_SIZE);

	// median

	long med_dims[DIMS];
	md_copy_dims(DIMS, med_dims, img_dims);
	med_dims[6] = median;

	complex float* med = md_calloc(DIMS, med_dims, CFL_SIZE);


	double start = timestamp();


	for (int n = 0; n < dims[6]; n++) {

		debug_printf(DP_INFO, "Frame: %d/%d\n", n, dims[6]);

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		for (int r = 0; r < dims[PHS1_DIM] / turns; r++) {

			if (-1 == adc_read(ifd, dims, pos, adc)) {

				debug_printf(DP_WARN, "Stopping.\n");
				break;
			}

			pos[6] = 0;

			if (!md_is_index(DIMS, pos, dims)) {

				debug_printf(DP_WARN, "Index out of bounds.\n");
				debug_print_dims(DP_WARN, DIMS, dims);
				debug_print_dims(DP_WARN, DIMS, pos);
			}


			pos[2] = pos[1];
			pos[1] = 0;

			md_copy_block(DIMS, pos, buf_dims, buf, adc_dims, adc, CFL_SIZE);

			pos[1] = 0;
			pos[2] = 0;
		}


		// coil compression

		if (0 == n)
			scc(cc_dims, cc, buf_dims, buf);

		md_ztenmulc(DIMS, bufT_dims, buf2, cc2_dims, cc, buf_dims, buf);

		// filter

		md_zmul2(DIMS, buf2_dims, buf2_strs, buf2, buf2_strs, buf2, filt_strs, filter);


		// reconstruct frame

		linop_adjoint(nufft_op, DIMS, coilim_dims, cimg, DIMS, buf2_dims, buf2);

		// RSS

		if (mcoil) {

			assert(img_dims[COIL_DIM] == coilim_dims[COIL_DIM]);
			md_copy(DIMS, coilim_dims, img, cimg, CFL_SIZE);

		} else {

			md_zrss(DIMS, coilim_dims, COIL_FLAG, img, cimg);
		}

		// Median filter

		if (1 != med_dims[6]) {

			pos[6] = n % med_dims[6];
			md_copy_block(DIMS, pos, med_dims, med, img_dims, img, CFL_SIZE);
			(geom ? md_geometric_medianz : md_medianz)(DIMS, 6, med_dims, img, med);
			pos[6] = 0;
		}

		pos[6] = n;
		pos[PHS1_DIM] = 0;
		pos[PHS2_DIM] = 0;
		md_copy_block(DIMS, pos, out_dims, out, img_dims, img, CFL_SIZE);
	}

	double end = timestamp();

	debug_printf(DP_INFO, "Time: %fs (%f frames/s)\n", end - start, dims[6] / (end - start));

	md_free(med);
	md_free(cc);
	md_free(adc);
	md_free(buf);
	md_free(img);
	md_free(cimg);
	md_free(filter);

	unmap_cfl(DIMS, out_dims, out);

	linop_free(nufft_op);
	exit(0);
}

