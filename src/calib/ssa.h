
#include <complex.h>

#include "misc/mri.h"
#include "manifold/delay_embed.h"
#include "manifold/manifold.h"


extern void ssa_fary(	const long cal_dims[DIMS],
			const long A_dims[2],
			const complex float* A,
			complex float* U,
			float* S_square,
			complex float* back,
			const struct delay_conf conf);

extern void nlsa_fary(	const long cal_dims[DIMS], 
			const long A_dims[2],
			const complex float* A,
			complex float* back,
			const struct delay_conf nlsa_conf, 
			const struct laplace_conf conf);


extern int detect_freq_EOF(const long dims[2], 
			complex float* EOF, 
			const float dt, 
			const float f,
			const float f_interval,
			const long max);

