#ifndef __BSFFT_H
#define __BSFFT_H

#define Mu_0 1.25663706212e-6
#define Hz_per_Tesla 42.577478518e6

struct linop_s *linop_bz_create(const long idims[4], const float vox[3]);
extern void biot_savart_fft(const long dims[4], const float vox[3], _Complex float *b, const _Complex float *j);
extern void jcylinder(const long dims[4], const float fov[3], const float R, const float h, const long d, _Complex float *out);

float bz_unit(const long dims[3], const float vox[3]);
void vox_to_fov(float fov[3], const long dims[3], const float vox[3]);
void fov_to_vox(float vox[3], const long dims[3], const float fov[3]);

#endif
