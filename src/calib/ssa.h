
#include <complex.h>

#include "misc/mri.h"
#include "manifold/delay_embed.h"


extern void ssa_fary(	const long cal_dims[DIMS],
			const long A_dims[2],
			const complex float* A,
			complex float* U,
			float* S_square,
			complex float* back,
			const struct delay_conf conf);


