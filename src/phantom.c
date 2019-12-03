/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019	     Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/phantom.h"
#include "simu/simulation.h"
#include "simu/bloch.h"



static const char usage_str[] = "<output>";
static const char help_str[] = "Image and k-space domain phantoms.";




int main_phantom(int argc, char* argv[])
{
	bool kspace = false;
	bool d3 = false;
	int sens = 0;
	int osens = -1;
	int xdim = -1;

	int geo = -1;
	enum ptype_e { SHEPPLOGAN, CIRC, TIME, HEART, SENS, GEOM, BART, T1T2 } ptype = SHEPPLOGAN;

	const char* traj = NULL;
	bool simulation = false;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 128;
	dims[1] = 128;
	dims[2] = 1;
	
	const struct opt_s opts[] = {

		OPT_INT('s', &sens, "nc", "nc sensitivities"),
		OPT_INT('S', &osens, "nc", "Output nc sensitivities"),
		OPT_SET('k', &kspace, "k-space"),
		OPT_STRING('t', &traj, "file", "trajectory"),
		OPT_SELECT('c', enum ptype_e, &ptype, CIRC, "()"),
		OPT_SELECT('m', enum ptype_e, &ptype, TIME, "()"),
		OPT_SELECT('G', enum ptype_e, &ptype, GEOM, "geometric object phantom"),
		OPT_SELECT('C', enum ptype_e, &ptype, HEART, "heart"),
		OPT_SELECT('T', enum ptype_e, &ptype, T1T2, "T1-T2 phantom"),
		OPT_SELECT('B', enum ptype_e, &ptype, BART, "BART letters"),
		OPT_INT('x', &xdim, "n", "dimensions in y and z"),
		OPT_INT('g', &geo, "n=1,2", "select geometry for object phantom"),
		OPT_SET('3', &d3, "3D"),
		OPT_SET('n', &simulation, "simulation"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if ((GEOM != ptype) && (-1 != geo)) {

		assert(SHEPPLOGAN == ptype);
		ptype = GEOM;
	}

	if ((GEOM == ptype) && (-1 == geo))
		geo = 1;


	if ((TIME == ptype) || (HEART == ptype))
		dims[TE_DIM] = 32;

	if (-1 != osens) {

		assert(SHEPPLOGAN == ptype);
		ptype = SENS;
		sens = osens;
	}

	if (-1 != xdim)
		dims[0] = dims[1] = xdim;

	if (d3)
		dims[2] = dims[0];


	long sdims[DIMS];
	long sstrs[DIMS];
	complex float* samples = NULL;

	if (NULL != traj) {

		kspace = true;

		samples = load_cfl(traj, DIMS, sdims);

		md_calc_strides(DIMS, sstrs, sdims, sizeof(complex float));

		dims[0] = 1;
		dims[1] = sdims[1];
		dims[2] = sdims[2];

		dims[TE_DIM] = sdims[TE_DIM];
	}
	else if(NULL == traj && simulation) //simulation default
		dims[TE_DIM] = 500;
	
	// values for simulation
	struct SimData sim_data;
				
	sim_data.seqData = seqData_defaults;
	sim_data.seqData.seq_type = 1;
	sim_data.seqData.TR = 0.0045;
	sim_data.seqData.TE = 0.00225;
	sim_data.seqData.rep_num = dims[TE_DIM];
	sim_data.seqData.spin_num = 1;
	sim_data.seqData.num_average_rep = 1; //need to be 1 in this implementation!!
	
	sim_data.voxelData = voxelData_defaults;
	
	sim_data.pulseData = pulseData_defaults;
	sim_data.pulseData.flipangle = 45.;
	sim_data.pulseData.RF_end = 0.0009;
	sim_data.gradData = gradData_defaults;
	sim_data.seqtmp = seqTmpData_defaults;
	
	if (sens > 0)
		dims[3] = sens;

	complex float* out;
	
	out = create_cfl(argv[1], DIMS, dims);
	md_zfill(DIMS, dims, out, 0.);
	
	md_clear(DIMS, dims, out, sizeof(complex float));
	
	if ( simulation && d3 ){
		debug_printf(DP_ERROR, "Numerical phantom does not work with 3D yet...\n");
		exit(0);
	}
	
	switch (ptype) {

	case SENS:

		assert(NULL == traj);
		assert(!kspace);

		calc_sens(dims, out);
		break;

	case HEART:

		assert(!d3);
		calc_heart(dims, out, kspace, sstrs, samples);

		break;

	case GEOM:

		if ((geo < 1) || (geo > 2))
			error("geometric phantom: invalid geometry");

		if (d3)
			error("geometric phantom: no 3D mode");

		calc_geo_phantom(dims, out, kspace, geo, sstrs, samples);
		break;

	case TIME:

		assert(!d3);
		calc_moving_circ(dims, out, kspace, sstrs, samples);
		break;

	case CIRC:

		calc_circ(dims, out, d3, kspace, sstrs, samples);
//			calc_ring(dims, out, kspace);
		break;

	case SHEPPLOGAN:
		
		calc_phantom(dims, out, d3, kspace, sstrs, samples);
		break;
        
	case T1T2:

		calc_phantom_t1t2(dims, out, d3, kspace, sstrs, samples);
		
		if (simulation)
			calc_simu_phantom(&sim_data, dims, out, kspace, sstrs, samples);
		
		break;
	
	case BART:

		calc_phantom_bart(dims, out, false, false, sstrs, samples);	
		break;
	}

	if (NULL != traj)
		free((void*)traj);
	
	if (NULL != samples)
		unmap_cfl(3, sdims, samples);

	unmap_cfl(DIMS, dims, out);
	
	return 0;
}


