
#ifndef DIMS
#define DIMS 16
#endif

extern void scale_psf_k(const long pat_dims[DIMS],
			_Complex float* pattern,
			const long ksp_dims[DIMS],
			_Complex float* kspace_data,
			const long trj_dims[DIMS],
			_Complex float* traj);


