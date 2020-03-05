
#include <stdlib.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"

#include "misc.h"

// Initialize dimensions and strides
void ds_init(struct ds_s* in, size_t size)
{
	md_select_dims(DIMS, ~TIME_FLAG, in->dims_singleFrame, in->dims_full);
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG, in->dims_output_singleFrame, in->dims_full);
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG|TIME_FLAG, in->dims_output, in->dims_full);
	md_select_dims(DIMS, ~SLICE_FLAG, in->dims_singlePart, in->dims_full);
	md_select_dims(DIMS, ~(TIME_FLAG|SLICE_FLAG), in->dims_singleFramePart, in->dims_full);

	md_calc_strides(DIMS, in->strs_full, in->dims_full, size);
	md_calc_strides(DIMS, in->strs_singleFrame, in->dims_singleFrame, size);
	md_calc_strides(DIMS, in->strs_singlePart, in->dims_singlePart, size);
	md_calc_strides(DIMS, in->strs_singleFramePart, in->dims_singleFramePart, size);
	md_calc_strides(DIMS, in->strs_output, in->dims_output, size);
	md_calc_strides(DIMS, in->strs_output_singleFrame, in->dims_output_singleFrame, size);

}

// Normalization of PSF and scaling of k-space
void scale_psf_k(struct ds_s* pat_s, complex float* pattern, struct ds_s* k_s, complex float* kspace_data, struct ds_s* traj_s, complex float* traj)
{
	/* PSF
	* Since for each frame we can have a different number of spokes,
	* some spoke-lines are empty in certain frames. To ensure
	* adequate normalization we have to calculate how many spokes are there
	* in each frame and build the inverse
	*
	* Basic idea:
	* Summation of READ_DIM and PHS1_DIM:
	* If the result is zero the spoke-line was empty
	*/

	long traj_dims2[DIMS]; // Squashed trajectory array
	md_copy_dims(DIMS, traj_dims2, traj_s->dims_full);
	traj_dims2[READ_DIM] = 1;
	traj_dims2[PHS1_DIM] = 1;
	complex float* traj2= md_alloc(DIMS, traj_dims2, CFL_SIZE);
	md_zrss(DIMS, traj_s->dims_full, READ_FLAG|PHS1_FLAG, traj2, traj);
	md_zdiv(DIMS, traj_dims2, traj2, traj2, traj2); // Normalize each non-zero element to one

	/* Sum the ones (non-zero elements) to get
	* number of spokes in each cardiac frame
	*/
	struct ds_s* no_spf_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_copy_dims(DIMS, no_spf_s->dims_full, traj_dims2);
	no_spf_s->dims_full[PHS2_DIM] = 1;
	ds_init(no_spf_s, CFL_SIZE);

	complex float* no_spf = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_clear(DIMS, no_spf_s->dims_full, no_spf, CFL_SIZE);
	md_zrss(DIMS, traj_dims2, PHS2_FLAG, no_spf, traj2);
	md_zspow(DIMS, no_spf_s->dims_full, no_spf, no_spf, 2); // no_spf contains the number of spokes in each frame and partition

	// Inverse (inv_no_spf contains inverse of number of spokes in each frame/partition)
	complex float* inv_no_spf = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_zfill(DIMS, no_spf_s->dims_full, inv_no_spf, 1.);
	md_zdiv(DIMS, no_spf_s->dims_full, inv_no_spf, inv_no_spf, no_spf);


	// Multiply PSF
	md_zmul2(DIMS, pat_s->dims_full, pat_s->strs_full, pattern, pat_s->strs_full, pattern, no_spf_s->strs_full, inv_no_spf);
	// 	dump_cfl("PSF", DIMS, pat_s->dims_full, pattern);

	/* k
	 * Scaling of k-space (depending on total [= all partitions] number of spokes per frame)
	 * Normalization is not performed here)
	 */

	// Sum spokes in all partitions
	complex float* no_spf_tot = md_alloc(DIMS, no_spf_s->dims_singlePart, CFL_SIZE);
	md_zsum(DIMS, no_spf_s->dims_full, SLICE_FLAG, no_spf_tot, no_spf);

	// Extract first frame
	complex float* no_sp_1stFrame_tot = md_alloc(DIMS, no_spf_s->dims_singleFramePart, CFL_SIZE);
	long posF[DIMS] = { 0 };
	md_copy_block(DIMS, posF, no_spf_s->dims_singleFramePart, no_sp_1stFrame_tot, no_spf_s->dims_singlePart, no_spf_tot, CFL_SIZE);

	complex float* ksp_scaleFactor = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_clear(DIMS, no_spf_s->dims_full, ksp_scaleFactor, CFL_SIZE);

	complex float* inv_no_spf_tot = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_zfill(DIMS, no_spf_s->dims_singlePart, inv_no_spf_tot, 1.);
	md_zdiv(DIMS, no_spf_s->dims_singlePart, inv_no_spf_tot, inv_no_spf_tot, no_spf_tot);
	md_zmul2(DIMS, no_spf_s->dims_full, no_spf_s->strs_full, ksp_scaleFactor, no_spf_s->strs_singlePart, inv_no_spf_tot, no_spf_s->strs_singleFramePart, no_sp_1stFrame_tot);

	md_zmul2(DIMS, k_s->dims_full, k_s->strs_full, kspace_data, k_s->strs_full, kspace_data, no_spf_s->strs_full, ksp_scaleFactor);

	free(no_spf_s);
	md_free(no_spf_tot);
	md_free(inv_no_spf_tot);
	md_free(ksp_scaleFactor);
	md_free(no_sp_1stFrame_tot);

	md_free(traj2);
	md_free(no_spf);
	md_free(inv_no_spf);
}


