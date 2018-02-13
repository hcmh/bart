
#define KERNEL_OVERSAMPLING 2.
#if 1
#define RKHSGRID
#endif

extern void comp_cholesky(int NN, double alpha, complex float* aha);
extern void calculate_kernelmatrix(const long kdims[8], complex float* kmat, int N, complex float* pos, const long dims[5], const complex float* kern);
extern complex float* comp_cardinal(const long kdims[3], long channels, const complex float* lhs, const complex float* aha);
extern void calculate_lhs(const long kdims[8], complex float* lhs, float npos[3], int N, complex float* pos, const long dims[5], const complex float* kern);
extern void calculate_lhsH(const long kdims[8], complex float* lhs, float npos[3], int N, complex float* pos, const long dims[5], const complex float* kern);
extern void calculate_diag(const long kdims[8], complex float* dia, const long dims[5], const complex float* kern);
extern complex float evaluate_kernel(int l, int k, const long dims[5], const complex float* kern, const float pos[3]);


