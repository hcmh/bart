/* Copyright 2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>

#include "simu/simulation.h"
#include "simu/signals.h"

#include "seq_model.h"




void looklocker_analytical(struct sim_data* simu_data, complex float* out)
{
	struct signal_model data;

	data.t1 = 1 / simu_data->voxel.r1;
	data.m0 = simu_data->voxel.m0;
	data.tr = simu_data->seq.tr;
	data.fa = simu_data->pulse.flipangle * M_PI / 180.;	//conversion to rad

	looklocker_model(&data, simu_data->seq.rep_num, out);
}


void IR_bSSFP_analytical(struct sim_data* simu_data, complex float* out)
{
	struct signal_model data;

	data.t1 = 1 / simu_data->voxel.r1;
	data.t2 = 1 / simu_data->voxel.r2;
	data.m0 = simu_data->voxel.m0;
	data.tr = simu_data->seq.tr;
	data.fa = simu_data->pulse.flipangle * M_PI / 180.;	//conversion to rad

	IR_bSSFP_model(&data, simu_data->seq.rep_num, out);
}


