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


static void help_seq(void)
{
	printf( "Sequence Simulation Parameter\n\n"
		"Typ:\t Define if analytical (1) or numerical simulation (0) should be performed \n"
		"#SEQ:\t Define sequence mode: \n"
		"\t\t\t0 = bSSFP[default]\n"
		"\t\t\t1 = invbSSFP\n"
		"\t\t\t2 = FLASH\n"
		"\t\t\t3 = pcbSSFP\n"
		"\t\t\t4 = inv. bSSFP without preparation\n"
		"\t\t\t5 = invFLASH\n"
		"\t\t\t6 = invpcbSSFP\n"
		"TR:\t Repetition time [s]\n"
		"TE:\t Echo time [s]\n"
		"Drf:\t Duration of RF pulse [s]\n"
		"FA:\t Flip angle of rf pulses [deg]\n"
	);
}


static bool opt_seq(void* ptr, char c, const char* optarg)
{
	(void) c;
	
	// Check if help function is called
	char rt[5];
	
	int ret = sscanf(optarg, "%4[^:]", rt);
	assert(1 == ret);
	
	if (strcmp(rt, "h") == 0) {

		help_seq();
		exit(0);
	} else {
		
		// Collect simulation data
		struct SimData* sim_data = ptr;

		ret = sscanf(optarg, "%d:%d:%f:%f:%f:%f",	&sim_data->seqData.analytical,
								&sim_data->seqData.seq_type, 
								&sim_data->seqData.TR, 
								&sim_data->seqData.TE, 
								&sim_data->pulseData.RF_end, 
								&sim_data->pulseData.flipangle);
		assert(6 == ret);
	}
	return false;
}



int main_phantom(int argc, char* argv[])
{
	bool kspace = false;
	bool d3 = false;
	int sens = 0;
	int osens = -1;
	int xdim = -1;

	int geo = -1;
	enum ptype_e { SHEPPLOGAN, CIRC, TIME, HEART, SENS, GEOM, STAR, BART, T1T2 } ptype = SHEPPLOGAN;

	const char* traj = NULL;
	bool simulation = false;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 128;
	dims[1] = 128;
	dims[2] = 1;
	
	// initalize values for simulation
	struct SimData sim_data;
	sim_data.seqData = seqData_defaults;
	sim_data.voxelData = voxelData_defaults;
	sim_data.pulseData = pulseData_defaults;
	sim_data.gradData = gradData_defaults;
	sim_data.seqtmp = seqTmpData_defaults;
	
	
	const struct opt_s opts[] = {

		OPT_INT('s', &sens, "nc", "nc sensitivities"),
		OPT_INT('S', &osens, "nc", "Output nc sensitivities"),
		OPT_SET('k', &kspace, "k-space"),
		OPT_STRING('t', &traj, "file", "trajectory"),
		OPT_SELECT('c', enum ptype_e, &ptype, CIRC, "()"),
		OPT_SELECT('a', enum ptype_e, &ptype, STAR, "()"),
		OPT_SELECT('m', enum ptype_e, &ptype, TIME, "()"),
		OPT_SELECT('G', enum ptype_e, &ptype, GEOM, "geometric object phantom"),
		OPT_SELECT('C', enum ptype_e, &ptype, HEART, "heart"),
		OPT_SELECT('T', enum ptype_e, &ptype, T1T2, "T1-T2 phantom"),
		OPT_SELECT('B', enum ptype_e, &ptype, BART, "BART letters"),
		OPT_INT('x', &xdim, "n", "dimensions in y and z"),
		OPT_INT('g', &geo, "n=1,2", "select geometry for object phantom"),
		OPT_SET('3', &d3, "3D"),
		OPT_SET('n', &simulation, "simulation"),
		{ 'P', true, opt_seq, &sim_data, "\tA:B:C:D:E:F\tParameters for Simulation <Typ:Seq:TR:TE:Drf:FA> (-Ph for help)" },
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
	else if (simulation)
		dims[TE_DIM] = 500;
	
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
	
	sim_data.seqData.rep_num = dims[TE_DIM];
	
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

	case STAR:

		(d3 ? calc_star3d : calc_star)(dims, out, kspace, sstrs, samples);
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
		
		calc_phantom_t1t2((simulation) ? &sim_data : NULL, dims, out, kspace, sstrs, samples);
		
		break;
	
	case BART:

		calc_bart(dims, out, kspace, sstrs, samples);	
		break;
	}

	if (NULL != traj)
		free((void*)traj);
	
	if (NULL != samples)
		unmap_cfl(3, sdims, samples);

	unmap_cfl(DIMS, dims, out);
	
	return 0;
}


