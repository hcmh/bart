
#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_zconvcorr_3D(_Complex float* dst, const _Complex float* src, const _Complex float* krn, long odims[3], long idims[3], long kdims[3], _Bool conv);
extern void cuda_zconvcorr_3D_CF(_Complex float* dst, const _Complex float* src, const _Complex float* krn, long odims[3], long idims[3], long kdims[3], _Bool conv);
extern void cuda_zconvcorr_3D_CF_TK(_Complex float* krn, const _Complex float* src, const _Complex float* out, long odims[3], long idims[3], long kdims[3], _Bool conv);
extern void cuda_zconvcorr_3D_CF_TI(_Complex float* im, const _Complex float* out, const _Complex float* krn, long odims[3], long idims[3], long kdims[3], _Bool conv);
extern void cuda_im2col(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5]);
extern void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, long odims[5], long idims[5], long kdims[5]);

#ifdef __cplusplus
}
#endif
