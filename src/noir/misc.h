
struct ds_s {

	long dims_full[DIMS];
	long dims_singleFrame[DIMS];
	long dims_singlePart[DIMS];
	long dims_singleFramePart[DIMS];
	long dims_output[DIMS];
	long dims_output_singleFrame[DIMS];


	long strs_full[DIMS];
	long strs_singleFrame[DIMS];
	long strs_singlePart[DIMS];
	long strs_singleFramePart[DIMS];
	long strs_output[DIMS];
	long strs_output_singleFrame[DIMS];

};


extern void ds_init(struct ds_s* dims, size_t size);

extern void scale_psf_k(struct ds_s* pat_s,
			_Complex float* pattern,
			struct ds_s* k_s,
			_Complex float* kspace_data,
			struct ds_s* traj_s,
			_Complex float* traj);


