#include <complex.h>
#include <math.h>

#include "linops/fmac.h"
#include "linops/someops.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"

#include "noncart/nufft.h"

#include "linops/linop.h"
#include "linops/fmac.h"


#include "misc.h"


struct network_data_s network_data_empty = {

	.ksp_dims = { 0 },
	.col_dims = { 0 },
	.psf_dims = { 0 },
	.img_dims = { 0 },
	.max_dims = { 0 },
	.cim_dims = { 0 },

	.filename_trajectory = NULL,
	.filename_pattern = NULL,
	.filename_kspace = NULL,
	.filename_coil = NULL,
	.filename_out = NULL,

	.kspace = NULL,
	.coil = NULL,
	.psf = NULL,
	.adjoint = NULL,
	.out = NULL,

	.create_out = false,
	.load_mem = false,

	.nufft_conf = &nufft_conf_defaults,
};


void load_network_data(struct network_data_s* nd) {

	nd->N = DIMS;
	nd->ND = DIMS;


	nd->coil = load_cfl(nd->filename_coil, DIMS, nd->col_dims);

	if (nd->load_mem) {

		complex float* tmp = anon_cfl("", DIMS, nd->col_dims);
		md_copy(DIMS, nd->col_dims, tmp, nd->coil, CFL_SIZE);
		unmap_cfl(DIMS, nd->col_dims, nd->coil);
		nd->coil = tmp;
	}

	nd->kspace = load_cfl(nd->filename_kspace, DIMS, nd->ksp_dims);

	md_copy_dims(DIMS, nd->max_dims, nd->ksp_dims);

	long pat_dims[DIMS];
	complex float* pattern;

	if (NULL != nd->filename_pattern) {

		pattern = load_cfl(nd->filename_pattern, DIMS, pat_dims);
		md_zmulc2(DIMS, nd->ksp_dims,
				MD_STRIDES(DIMS, nd->ksp_dims, CFL_SIZE), nd->kspace,
				MD_STRIDES(DIMS, nd->ksp_dims, CFL_SIZE), nd->kspace,
				MD_STRIDES(DIMS, pat_dims, CFL_SIZE), pattern);
	} else {

		md_select_dims(DIMS, ~(COIL_FLAG), pat_dims, nd->ksp_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, nd->ksp_dims, COIL_FLAG, pattern, nd->kspace);
	}

	md_copy_dims(DIMS, nd->max_dims, nd->ksp_dims);
	md_copy_dims(5, nd->max_dims, nd->col_dims);

	md_select_dims(DIMS, ~MAPS_FLAG, nd->cim_dims, nd->max_dims);
	md_select_dims(DIMS, ~COIL_FLAG, nd->img_dims, nd->max_dims);

	nd->adjoint = md_alloc(DIMS, nd->img_dims, CFL_SIZE);

	unsigned long cim_flags = md_nontriv_dims(DIMS, nd->cim_dims);
	unsigned long img_flags = md_nontriv_dims(DIMS, nd->img_dims);
	unsigned long col_flags = md_nontriv_dims(DIMS, nd->col_dims);

	if (NULL == nd->filename_trajectory) {

		const struct linop_s* lop_frw = linop_fmac_create(DIMS, nd->max_dims, ~cim_flags, ~img_flags, ~col_flags, nd->coil);
		lop_frw = linop_chain_FF(lop_frw, linop_resize_center_create(DIMS, nd->ksp_dims, nd->cim_dims));
		lop_frw = linop_chain_FF(lop_frw, linop_fftc_create(DIMS, nd->ksp_dims, FFT_FLAGS));

		linop_adjoint(lop_frw, DIMS, nd->img_dims, nd->adjoint, DIMS, nd->ksp_dims, nd->kspace);

		linop_free(lop_frw);

		long pat_strs[DIMS];
		md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

		md_copy_dims(DIMS, nd->psf_dims, pat_dims);
		for (int i = 0; i < DIMS; i++) {

			long pat_strs2[DIMS];
			md_copy_dims(DIMS, pat_strs2, pat_strs);
			pat_strs2[i] = 0;

			if ( (1 == nd->psf_dims[i]) || md_compare2(DIMS, nd->psf_dims, pat_strs2, pattern, pat_strs, pattern, CFL_SIZE) )
				nd->psf_dims[i] = 1;
		}

		assert(md_check_equal_dims(DIMS, nd->psf_dims, nd->max_dims, md_nontriv_dims(DIMS, nd->psf_dims)));


		nd->psf = md_alloc(DIMS, nd->psf_dims, CFL_SIZE);

		md_resize(DIMS, nd->psf_dims, nd->psf, pat_dims, pattern, CFL_SIZE);
		md_zmulc(DIMS, nd->psf_dims, nd->psf, nd->psf, nd->psf);

	} else {

		complex float* traj = NULL;
		long trj_dims[DIMS];
		traj = load_cfl(nd->filename_trajectory, DIMS, trj_dims);

		const struct linop_s* fft_op = nufft_create2(DIMS, nd->ksp_dims, nd->cim_dims, trj_dims, traj, pat_dims, pattern, NULL, NULL, *(nd->nufft_conf));

		if (DIMS + 1 != nufft_get_psf_dims(fft_op, DIMS + 1, nd->psf_dims))
			assert(0);

		nd->ND = DIMS + 1;

		nd->psf = md_alloc(DIMS + 1, nd->psf_dims, CFL_SIZE);
		nufft_get_psf(fft_op, DIMS + 1, nd->psf_dims, nd->psf);

		const struct linop_s* maps_op = linop_fmac_create(DIMS, nd->max_dims, ~cim_flags, ~img_flags, ~col_flags, nd->coil);
		const struct linop_s* lop_frw = linop_chain_FF(maps_op, fft_op);

		linop_adjoint(lop_frw, DIMS, nd->img_dims, nd->adjoint, DIMS, nd->ksp_dims, nd->kspace);

		linop_free(lop_frw);

		unmap_cfl(DIMS, trj_dims, traj);
	}

	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, nd->ksp_dims, nd->kspace);
	nd->kspace = NULL;

	if (nd->create_out) {

		nd->out = create_cfl(nd->filename_out, DIMS, nd->img_dims);
	} else {

		long idims_file[DIMS];
		nd->out = load_cfl(nd->filename_out, DIMS, idims_file);
		assert(md_check_equal_dims(DIMS, nd->img_dims, idims_file, ~0));

		if (nd->load_mem) {

			complex float* out_tmp = anon_cfl("", DIMS, nd->img_dims);
			md_copy(DIMS, nd->img_dims, out_tmp, nd->out, CFL_SIZE);
			unmap_cfl(DIMS, nd->img_dims, nd->out);
			nd->out = out_tmp;
		}
	}
}

void free_network_data(struct network_data_s* nd)
{
	md_free(nd->psf);
	md_free(nd->adjoint);

	unmap_cfl(DIMS, nd->col_dims, nd->coil);
	unmap_cfl(DIMS, nd->img_dims, nd->out);
}