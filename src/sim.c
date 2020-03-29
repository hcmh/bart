
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
#include "simu/seq_model.h"
#include "simu/bloch.h"
#include "simu/sim_para.h"


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

			ret = sscanf(optarg, "%d:%d:%f:%f:%f:%f:%d",	&sim_data->seq.analytical,
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
	int nbf = 1;
	bool ode = false;

	// initalize values for simulation
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.tmp = simdata_tmp_defaults;


	const struct opt_s opts[] = {

		OPT_INT('n', &nbf, "nbf", "No. Basis Functions"),
		OPT_SET('o', &ode, "ODE based simulation [Default: OBS]"),
		{ 'P', true, opt_seq, &sim_data, "\tA:B:C:D:E:F:G\tParameters for Simulation <Typ:Seq:tr:te:Drf:FA:#tr> (-Ph for help)" },
	};


	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(nbf <= MAX_REF_VALUES);

	// Preparation for realistic sequence simulation
	sim_data.seq.prep_pulse_length = sim_data.seq.tr/2.;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = sim_data.seq.rep_num;
	dims[COEFF_DIM] = nbf;

	long dims1[DIMS];
	md_select_dims(DIMS, TE_FLAG, dims1, dims);

	complex float* basis_functions = create_cfl(argv[1], DIMS, dims);
	md_zfill(DIMS, dims, basis_functions, 1.0);

	// Apply simulation to all geometrical structures to determine time evolution of signal
	#pragma omp parallel for
	for (int j = 0; j < dims[COEFF_DIM]; j++) {

		struct sim_data data = sim_data;

		data.voxel.r1 = 1 / tissue_ref_para[j].t1;
		data.voxel.r2 = 1 / tissue_ref_para[j].t2;
		data.voxel.m0 = tissue_ref_para[j].m0;

		if (data.seq.analytical) {

			complex float* signal = md_alloc(DIMS, dims1, CFL_SIZE);

			if (2 == data.seq.seq_type || 5 == data.seq.seq_type)
				looklocker_analytical(&data, signal);
			else if (1 == data.seq.seq_type)
				IR_bSSFP_analytical(&data, signal);
			else
				debug_printf(DP_ERROR, "Analytical function of desired sequence is not provided.\n");


			for (int t = 0; t < dims[TE_DIM]; t++) 
				basis_functions[j * dims[TE_DIM] + t] = signal[t];


		} 
		else { // TODO: change to complex floats!!

			float mxy_sig[data.seq.rep_num / data.seq.num_average_rep][3];
			float sa_r1_sig[data.seq.rep_num / data.seq.num_average_rep][3];
			float sa_r2_sig[data.seq.rep_num / data.seq.num_average_rep][3];
			float sa_m0_sig[data.seq.rep_num / data.seq.num_average_rep][3];

			if (ode)
				ode_bloch_simulation3(&data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);	// ODE simulation
			else
				matrix_bloch_simulation(&data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig);	// OBS simulation, does not work with hard-pulses!

			for (int t = 0; t < dims[TE_DIM]; t++) 
				basis_functions[j * dims[TE_DIM] + t] = mxy_sig[t][1] + mxy_sig[t][0] * I;
		}
	}

	unmap_cfl(DIMS, dims, basis_functions );

	return 0;
}


