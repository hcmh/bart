
#include <math.h>
#include <stdio.h>
#include <complex.h>
#include <stdbool.h>

#include "simu/slice_profile.h"
#include "simu/sim_matrix.h"
#include "simu/pulse.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "utest.h"


static bool test_slice_profile(void)
{
	enum { number_isochromats = 10 };

	long slcprfl_dims[DIMS];
	md_set_dims(DIMS, slcprfl_dims, 1);
	slcprfl_dims[READ_DIM] = number_isochromats;

	complex float* sliceprofile = md_alloc(DIMS, slcprfl_dims, CFL_SIZE);

	// FIXME: Extend also to other rf pulse shapes
	estimate_slice_profile(DIMS, slcprfl_dims, sliceprofile);

	float reference[number_isochromats] = {	0.023543, 0.083183,
						0.192155, 0.348666,
						0.532440, 0.710757,
						0.852502, 0.942181,
						0.984873, 0.998577};

	float err = 0.00001;

	for (int i = 0; i < number_isochromats; i++) {

		if (err < fabsf(reference[i] - crealf(sliceprofile[i])))
			return 0;
	}
	return 1;
}

UT_REGISTER_TEST(test_slice_profile);
