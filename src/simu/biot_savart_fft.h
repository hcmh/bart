#ifndef __BSFFT_H
#define __BSFFT_H
struct linop_s* linop_bz_create(const long idims[4], const float fov[3]);
extern void biot_savart_fft(const long *dims, const float *fov, _Complex float* j, _Complex float* b);
extern void jcylinder(const long* dims, const float *fov, const float R, const float h, const long d, _Complex float* out);
float bz_unit(const long N, const float* fov);

#endif
