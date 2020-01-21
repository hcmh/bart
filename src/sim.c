
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
#include "simu/sim_matrix.h"
#include "simu/seq_model.h"
#include "simu/bloch.h"



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
		"TR:\t Repetition time [s]\n"
		"TE:\t Echo time [s]\n"
		"Drf:\t Duration of RF pulse [s]\n"
		"FA:\t Flip angle of rf pulses [deg]\n"
		"#TR:\t Number of repetitions\n"
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

		ret = sscanf(optarg, "%d:%d:%f:%f:%f:%f:%d",	&sim_data->seqData.analytical,
								&sim_data->seqData.seq_type, 
								&sim_data->seqData.TR, 
								&sim_data->seqData.TE, 
								&sim_data->pulseData.RF_end, 
								&sim_data->pulseData.flipangle,
								&sim_data->seqData.rep_num);
		assert(7 == ret);
	}
	return false;
}



int main_sim(int argc, char* argv[])
{
	int nbf = 8;
	
	// initalize values for simulation
	struct SimData sim_data;
	sim_data.seqData = seqData_defaults;
	sim_data.voxelData = voxelData_defaults;
	sim_data.pulseData = pulseData_defaults;
	sim_data.gradData = gradData_defaults;
	sim_data.seqtmp = seqTmpData_defaults;
	
	
	const struct opt_s opts[] = {
		
		OPT_INT('n', &nbf, "nbf", "No. Basis Functions"),
		{ 'P', true, opt_seq, &sim_data, "\tA:B:C:D:E:F:G\tParameters for Simulation <Typ:Seq:TR:TE:Drf:FA:#TR> (-Ph for help)" },
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	
	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = sim_data.seqData.rep_num;
	dims[COEFF_DIM] = nbf;
	
	long dims1[DIMS];
	md_select_dims(DIMS, TE_FLAG, dims1, dims);
	
	complex float* basis_functions = create_cfl(argv[1], DIMS, dims);
	md_zfill(DIMS, dims, basis_functions, 1.0);
	
	
	// Choose more realistic values (rand, manual and homogeneously distributed)
	float t1[nbf];
	float t2[nbf];
	
	for (int i = 0; i < nbf; i++) {
		t1[i] = 3. - i * (3. - 0.1)/(float) nbf;
		t2[i] = 1. - i * (1. - 0.04)/(float) nbf;
	}

	float m0 = 1.;
	
	// Apply simulation to all geometrical structures to determine time evolution of signal
	#pragma omp parallel for
	for (int j = 0; j < dims[COEFF_DIM]; j++) {
		
		struct SimData data = sim_data;

		data.voxelData.r1 = 1/t1[j];
		data.voxelData.r2 = 1/t2[j];
		data.voxelData.m0 = m0;
		
		if (data.seqData.analytical) {
			
			complex float* signal = md_alloc(DIMS, dims1, CFL_SIZE);
			
			if (5 == data.seqData.seq_type)
				
				looklocker_analytical(&data, signal);
			
			else if (1 == data.seqData.seq_type)
				
				IR_bSSFP_analytical(&data, signal);
			else
				
				debug_printf(DP_ERROR, "Analytical function of desired sequence is not provided.\n");
			
			
			for (int t = 0; t < dims[TE_DIM]; t++) 
				basis_functions[j * dims[TE_DIM] + t] = signal[t];
			
			
		} else { // TODO: change to complex floats!!
			
			float mxySig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saR1Sig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saR2Sig[data.seqData.rep_num / data.seqData.num_average_rep][3];
			float saDensSig[data.seqData.rep_num / data.seqData.num_average_rep][3];

			ode_bloch_simulation3(&data, mxySig, saR1Sig, saR2Sig, saDensSig);	// ODE simulation
// 			matrix_bloch_simulation(&data, mxySig, saR1Sig, saR2Sig, saDensSig);	// OBS simulation, does not work with hard-pulses!
			
			for (int t = 0; t < dims[TE_DIM]; t++) 
				basis_functions[j * dims[TE_DIM] + t] = mxySig[t][1] + mxySig[t][0] * I;
		}
	}
	
	unmap_cfl(DIMS, dims, basis_functions );
	
	return 0;
}


