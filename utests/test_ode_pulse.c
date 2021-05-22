/* Author:
 * 	2018-2021 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */

#include <math.h>
#include <stdio.h>

#include "num/ode.h"
#include "simu/bloch.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/pulse.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "utest.h"

// For visualization of pulse shape
static bool test_sinc_function(void)
{
	struct simdata_pulse pulse = simdata_pulse_defaults;

	pulse_create(&pulse, 0., 0.009, 90., 0., 4., 0.46);

	float pulse_length = pulse.rf_end - pulse.rf_start;
	float samples = 1000;
	float dt = pulse_length / samples;

	long dims[DIMS];

	md_set_dims(DIMS, dims, 1);
	dims[READ_DIM] = samples;

	complex float* storage = md_alloc(DIMS, dims, CFL_SIZE);


	for (int i = 0; i < samples ; i ++)
		storage[i] = pulse_sinc(&pulse, pulse.rf_start + i * dt );
#if 0
	dump_cfl("_pulse_shape", DIMS, dims, storage);
#endif
	md_free(storage);
	return 1;
}

UT_REGISTER_TEST(test_sinc_function);


static bool test_rf_pulse_ode(void)
{
	long dim[DIMS] = { [0 ... DIMS - 1] = 1 };

	dim[0] = 10;
	dim[1] = 10;

	float trfmin = 0.0001;
	float trfmax = 0.1;
	float amin = 1.;
	float amax = 180.;

	for (int i = 0; i < dim[0]; i++ )
		for (int j = 0; j < dim[1]; j++ ) {

			float trf = (trfmin + (float)i/((float)dim[0] - 1.) * (trfmax - trfmin));
			float angle = (amin + (float)j/((float)dim[1] - 1.) * (amax - amin));

			struct sim_data data;

			data.seq = simdata_seq_defaults;
			data.seq.seq_type = 1;
			data.seq.tr = 10;
			data.seq.te = 5;
			data.seq.rep_num = 1;
			data.seq.spin_num = 1;
			data.seq.num_average_rep = 1;

			data.voxel = simdata_voxel_defaults;
			data.voxel.r1 = 0.;
			data.voxel.r2 = 0.;
			data.voxel.m0 = 1;
			data.voxel.w = 0;

			data.pulse = simdata_pulse_defaults;
			data.pulse.flipangle = angle;
			data.pulse.rf_end = trf;
			data.grad = simdata_grad_defaults;
			data.tmp = simdata_tmp_defaults;

			pulse_create(&data.pulse, 0, trf, angle, 0, 4., 0.46);

			float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } }; //xp[P + 2][N]

			float h = 10E-5;
			float tol = 10E-6;
			int N = 3;
			int P = 2;

			start_rf_pulse( &data, h, tol, N, P, xp);

			float sim_angle = 0.;

			if (xp[0][2] >= 0) { //case of FA <= 90°

				if (data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = asinf(xp[0][1] / data.voxel.m0) / M_PI * 180.;
				else
					sim_angle = asinf(xp[0][1] / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2])) / M_PI * 180.;
			}
			else { //case of FA > 90°

				if (data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = acosf( fabsf(xp[0][1]) / data.voxel.m0 ) / M_PI * 180. + 90.;
				else
					sim_angle = acosf( fabsf(xp[0][1]) / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2]) ) / M_PI * 180. + 90.;
			}

			float err = fabsf( data.pulse.flipangle - sim_angle );

			if (err > 10E-2) {

				debug_printf(DP_WARN, "Error in rf-pulse test!\n see -> utests/test_ode_simu.c\n");
				return 0;
			}
		}
	return 1;
}

UT_REGISTER_TEST(test_rf_pulse_ode);




static bool test_rf_pulse_matexp(void)
{
	long dim[DIMS] = { [0 ... DIMS - 1] = 1 };
	dim[0] = 10;
	dim[1] = 10;

	float trfmin = 0.0001;
	float trfmax = 0.1;
	float amin = 1.;
	float amax = 180.;

	for (int i = 0; i < dim[0]; i++ )
		for (int j = 0; j < dim[1]; j++ ) {

			float trf = (trfmin + (float)i/((float)dim[0] - 1.) * (trfmax - trfmin));
			float angle = (amin + (float)j/((float)dim[1] - 1.) * (amax - amin));

			struct sim_data data;

			data.seq = simdata_seq_defaults;
			data.seq.seq_type = 1;
			data.seq.tr = 10;
			data.seq.te = 5;
			data.seq.rep_num = 1;
			data.seq.spin_num = 1;
			data.seq.num_average_rep = 1;

			data.voxel = simdata_voxel_defaults;
			data.voxel.r1 = 0.;
			data.voxel.r2 = 0.;
			data.voxel.m0 = 1;
			data.voxel.w = 0;

			data.pulse = simdata_pulse_defaults;
			data.pulse.flipangle = angle;
			data.pulse.rf_end = trf;
			data.grad = simdata_grad_defaults;
			data.tmp = simdata_tmp_defaults;

			data.pulse.pulse_applied = true;

			pulse_create(&data.pulse, 0, trf, angle, 0, 4., 0.46);

			enum { N = 13 };

			float x0[N] = { 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. };
			float x1[N];


			//Starting first pulse
			float m[N][N];
			mat_exp_simu( N, data.pulse.rf_end, m, &data);

			for (int i = 0; i < N; i++) {

				x1[i] = 0.;

				for (int j = 0; j < N; j++)
					x1[i] += m[j][i] * x0[j];
			}


			float sim_angle = 0.;

			if (x1[2] >= 0) { //case of FA <= 90°

				if(data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = asinf(x1[1] / data.voxel.m0) / M_PI * 180.;
				else
					sim_angle = asinf(x1[1] / sqrtf(x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2])) / M_PI * 180.;
			}
			else {//case of FA > 90°

				if (data.voxel.r1 != 0 && data.voxel.r2 != 0)//for relaxation case
					sim_angle = acosf( fabsf(x1[1]) / data.voxel.m0 ) / M_PI * 180. + 90.;
				else
					sim_angle = acosf( fabsf(x1[1]) / sqrtf(x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2]) ) / M_PI * 180. + 90.;
			}


			float err = fabsf(data.pulse.flipangle - sim_angle);

			if (10E-2 < err) {

				debug_printf(DP_WARN, "Error in mat rf-pulse test!\n see -> utests/test_ode_simu.c\n");
				return 0;
			}
		}
	return 1;
}

UT_REGISTER_TEST(test_rf_pulse_matexp);