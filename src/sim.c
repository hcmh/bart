
#include <math.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <string.h>

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
#include "simu/sim_matrix.h"
#include "simu/bloch.h"
#include "simu/signals.h"


static const char usage_str[] = "<OUT:basis-functions>";
static const char help_str[] = "Basis function based simulation tool.";


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
		"tr:\t Repetition time [s]\n"
		"te:\t Echo time [s]\n"
		"Drf:\t Duration of RF pulse [s]\n"
		"FA:\t Flip angle of rf pulses [deg]\n"
		"#tr:\t Number of repetitions\n"
	);
}

static void sim_to_signal_struct (struct signal_model* signal_data, struct sim_data* sim_data)
{
	signal_data->t1 = 1 / sim_data->voxel.r1;
	signal_data->t2 = 1 / sim_data->voxel.r2;
	signal_data->m0 = sim_data->voxel.m0;

	signal_data->fa = sim_data->pulse.flipangle * M_PI / 180.;
	signal_data->tr = sim_data->seq.tr;
	signal_data->te = sim_data->seq.te;

	signal_data->ir = (1 == sim_data->seq.seq_type || 5 == sim_data->seq.seq_type) ? true : false;
}

static bool opt_seq(void* ptr, char c, const char* optarg)
{
	// Check if help function is called
	char rt[5];
	
	switch (c) {

	case 'P': {
		
		int ret = sscanf(optarg, "%7[^:]", rt);
		assert(1 == ret);

		if (strcmp(rt, "h") == 0) {

			help_seq();
			exit(0);
		}
		else {

			// Collect simulation data
			struct sim_data* sim_data = ptr;

			ret = sscanf(optarg, "%d:%d:%f:%f:%f:%f:%d",	
									&sim_data->seq.analytical,
									&sim_data->seq.seq_type, 
									&sim_data->seq.tr, 
									&sim_data->seq.te, 
									&sim_data->pulse.rf_end, 
									&sim_data->pulse.flipangle,
									&sim_data->seq.rep_num);
			assert(7 == ret);
		}
		break;
	}
	}
	return false;
}


int main_sim(int argc, char* argv[])
{
	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };

	bool ode = false;

	float T1[3] = { 0.5, 4., 10 };
	float T2[3] = {  0.05,  0.5, 10 };

	// initalize values for simulation
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;


	const struct opt_s opts[] = {

		OPT_SET('o', &ode, "ODE based simulation [Default: OBS]"),
		OPT_FLVEC3('1', &T1, "min:max:N", "range of T1s"),
		OPT_FLVEC3('2', &T2, "min:max:N", "range of T2s"),
		{ 'P', true, opt_seq, &sim_data, "\tA:B:C:D:E:F:G\tParameters for Simulation <Typ:Seq:tr:te:Drf:FA:#tr> (-Ph for help)" },
	};


	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	// Pass pre defined data
	dims[TE_DIM] = sim_data.seq.rep_num;

	sim_data.seq.prep_pulse_length = sim_data.pulse.rf_end;


	// Prepare analytical case
	struct signal_model parm;
	
	if (sim_data.seq.analytical) {

		switch (sim_data.seq.seq_type) {
		
		case 1: parm = signal_IR_bSSFP_defaults; break;
		case 2: parm = signal_looklocker_defaults; break;
		case 5: parm = signal_looklocker_defaults; break;

		default: error("sequence type not supported");
		}

		sim_to_signal_struct(&parm, &sim_data);
	}


	// Prepare multi relaxation parameter simulation
	dims[COEFF_DIM] = truncf(T1[2]);
	dims[COEFF2_DIM] = truncf(T2[2]);
	
	if ((dims[TE_DIM] < 1) || (dims[COEFF_DIM] < 1) || (dims[COEFF2_DIM] < 1))
		error("invalid parameter range");


	complex float* signals = create_cfl(argv[1], DIMS, dims);


	long dims1[DIMS];
	md_select_dims(DIMS, TE_FLAG, dims1, dims);

	long pos[DIMS] = { 0 };
	int N = dims[TE_DIM];

	do {
		sim_data.voxel.r1 = 1. / ( T1[0] + (T1[1] - T1[0]) / T1[2] * (float)pos[COEFF_DIM] );
		sim_data.voxel.r2 = 1. / ( T2[0] + (T2[1] - T2[0]) / T2[2] * (float)pos[COEFF2_DIM] );
		sim_data.voxel.m0 = 1.;

		complex float out[N];

		if (sim_data.seq.analytical) {

			parm.t1 = 1 / sim_data.voxel.r1;
			parm.t2 = 1 / sim_data.voxel.r2;
			parm.m0 = sim_data.voxel.m0;

			switch (sim_data.seq.seq_type) {

			case 1: IR_bSSFP_model(&parm, N, out); break;
			case 2: looklocker_model(&parm, N, out); break;
			case 5: looklocker_model(&parm, N, out); break;

			default: assert(0);
			}
		} 
		else
			bloch_simulation(&sim_data, N, out, ode);


		md_copy_block(DIMS, pos, dims, signals, dims1, out, CFL_SIZE);

	} while(md_next(DIMS, dims, ~TE_FLAG, pos));

	unmap_cfl(DIMS, dims, signals);

	return 0;
}


