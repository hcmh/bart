extern void compute_zero_filled(	const long udims[5], _Complex float * out,
					const long kdims[5], const _Complex float* kspace, const _Complex float* coil, const long pdims[5], const _Complex float* pattern);
extern void compute_scale_max_abs(const long dims[5], _Complex float* scaling, const _Complex float * u0);
extern void normalize_by_scale(const long dims[5], const _Complex float* scaling, _Complex float* out, const _Complex float* in);
extern void renormalize_by_scale(const long dims[5], const _Complex float* scaling, _Complex float* out, const _Complex float* in);