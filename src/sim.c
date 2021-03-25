
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
#include "simu/polar_angles.h"
#include "simu/variable_flipangles.h"


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
		"\t\t\t4 = antiperiodic HSFP\n"
		"\t\t\t5 = invFLASH\n"
		"\t\t\t6 = invpcbSSFP\n"
		"tr:\t Repetition time [s]\n"
		"te:\t Echo time [s]\n"
		"Drf:\t Duration of RF pulse [s]\n"
		"FA:\t Flip angle of rf pulses [deg]\n"
		"#tr:\t Number of repetitions\n"
		"dw:\t off-resonance\n"
		"Dinv:\t inversion time\n"
		"BWTP:\t BandWidth Time Product of pulses\n"
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


// FIXME: Separate simulation parameter -P ..:... for pulse, -S ...:... for sequence, etc. to clean up interface
static bool opt_seq(void* ptr, char c, const char* optarg)
{
	// Check if help function is called
	char rt[5];
	
	switch (c) {

	case 'P': {
		
		int ret = sscanf(optarg, "%10[^:]", rt);
		assert(1 == ret);

		if (strcmp(rt, "h") == 0) {

			help_seq();
			exit(0);
		}
		else {

			// Collect simulation data
			struct sim_data* sim_data = ptr;

			ret = sscanf(optarg, "%d:%d:%f:%f:%f:%f:%d:%f:%f:%f",
									&sim_data->seq.analytical,
									&sim_data->seq.seq_type, 
									&sim_data->seq.tr, 
									&sim_data->seq.te, 
									&sim_data->pulse.rf_end, 
									&sim_data->pulse.flipangle,
									&sim_data->seq.rep_num,
									&sim_data->voxel.w,
									&sim_data->seq.inversion_pulse_length,
									&sim_data->pulse.bwtp);
			assert(10 == ret);
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

	const char* z_component = NULL;
	const char* radial_component = NULL;

	const struct opt_s opts[] = {

		OPT_SET(	'o', 	&ode, 			"ODE based simulation [Default: OBS]"),
		OPT_FLVEC3(	'1',	&T1, 			"min:max:N", "range of T1s"),
		OPT_FLVEC3(	'2',	&T2, 			"min:max:N", "range of T2s"),
		OPT_STRING(	'z',	&z_component,		"", "Output z component"),
		OPT_STRING(	'r',	&radial_component,	"", "Output radial component"),
		OPTL_SET(0, "look-locker-assumption", &sim_data.seq.look_locker_assumptions, "Turn on Look-Locker Assumption?"),
		{ 'P', NULL, true, opt_seq, &sim_data, "\tA:B:C:D:E:F:G:H:I:J\tParameters for Simulation <Typ:Seq:tr:te:Drf:FA:#tr:dw:Dinv:BWTP> (-Ph for help)" },
	};


	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	// Hidden option for MOLLI
#if 0
	
	sim_data.seq.molli_break = 200; // unit [#tr]
	sim_data.seq.molli_measure = 100; // unit [#tr]. Number of TR after which a break for $molli_break occurrs

	// Number of repetitions needs to be multiple of molli_measure number 
	assert(floor(sim_data.seq.rep_num/sim_data.seq.molli_measure) == sim_data.seq.rep_num/sim_data.seq.molli_measure);
	assert(5 == sim_data.seq.seq_type);

	if ( (0. != sim_data.seq.molli_break) && (!ode) )
		error("MOLLI mode only for ODE simulation yet!");
#endif

	// Pass pre defined data

	dims[TE_DIM] = sim_data.seq.rep_num;

	// TODO: Fix pass preperation time to sim tool
	if (5 == sim_data.seq.seq_type)
		sim_data.seq.prep_pulse_length = 0.0001;	// match analytical Look-Locker model
	else
		sim_data.seq.prep_pulse_length = sim_data.pulse.rf_end;

	// Prepare analytical case

	struct signal_model parm;
	
	if (sim_data.seq.analytical) {

		switch (sim_data.seq.seq_type) {
		
		case 1: parm = signal_IR_bSSFP_defaults; break;
		case 2: parm = signal_looklocker_defaults; break;
		case 4: parm = signal_hsfp_defaults; break;
		case 5: parm = signal_looklocker_defaults; break;

		default: error("sequence type not supported");
		}

		sim_to_signal_struct(&parm, &sim_data);
	}

	// Import variable flipangle file

	if (4 == sim_data.seq.seq_type) {

		assert(ode);

		if (1 == sim_data.seq.analytical)
			assert(N_PA_ANTIHSFP >= sim_data.seq.rep_num);

		long vfa_dims[DIMS];
		md_set_dims(DIMS, vfa_dims, 1);
		vfa_dims[READ_DIM] = sim_data.seq.rep_num;

		debug_print_dims(DP_DEBUG2, DIMS, vfa_dims);

		sim_data.seq.variable_fa = md_alloc(DIMS, vfa_dims, CFL_SIZE);

		get_antihsfp_fa(sim_data.seq.rep_num, sim_data.seq.variable_fa);
	}

	// Prepare multi relaxation parameter simulation

	dims[COEFF_DIM] = truncf(T1[2]);
	dims[COEFF2_DIM] = truncf(T2[2]);
	
	if ((dims[TE_DIM] < 1) || (dims[COEFF_DIM] < 1) || (dims[COEFF2_DIM] < 1))
		error("invalid parameter range");

	// Allocate memory for magnetization components

	complex float* x_magnetization = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y_magnetization = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* z_magnetization = md_alloc(DIMS, dims, CFL_SIZE);

	// Output z check up's

	if (NULL != z_component || NULL != radial_component) {

		if (sim_data.seq.analytical && 4 == sim_data.seq.seq_type)
			debug_printf(DP_WARN, "Analytical HSFP model only holds for radial component! Output as signal!\n");

		assert(!sim_data.seq.analytical);
	}

	long dims1[DIMS];
	md_select_dims(DIMS, TE_FLAG, dims1, dims);

	long pos[DIMS] = { 0 };
	int N = dims[TE_DIM];

	do {
		sim_data.voxel.r1 = 1. / ( T1[0] + (T1[1] - T1[0]) / T1[2] * (float)pos[COEFF_DIM] );
		sim_data.voxel.r2 = 1. / ( T2[0] + (T2[1] - T2[0]) / T2[2] * (float)pos[COEFF2_DIM] );
		sim_data.voxel.m0 = 1.;

		complex float out_x[N], out_y[N], out_z[N];

		if (sim_data.seq.analytical) {

			parm.t1 = 1 / sim_data.voxel.r1;
			parm.t2 = 1 / sim_data.voxel.r2;
			parm.m0 = sim_data.voxel.m0;

			switch (sim_data.seq.seq_type) {

			case 1: IR_bSSFP_model(&parm, N, out_x); break;
			case 2: looklocker_model(&parm, N, out_x); break;
			case 4: hsfp_simu(&parm, N, polarangles_antihsfp, out_x, false); break;
			case 5: looklocker_model(&parm, N, out_x); break;

			default: assert(0);
			}
		} 
		else
			bloch_simulation(&sim_data, N, out_x, out_y, out_z, ode);

		md_copy_block(DIMS, pos, dims, x_magnetization, dims1, out_x, CFL_SIZE);
		// FIXME: Mapping NULL -> NULL ugly for analytical case...
		md_copy_block(DIMS, pos, dims, y_magnetization, dims1, out_y, CFL_SIZE);
		md_copy_block(DIMS, pos, dims, z_magnetization, dims1, out_z, CFL_SIZE);

	} while(md_next(DIMS, dims, ~TE_FLAG, pos));

	// Determine signal

	complex float* signals = create_cfl(argv[1], DIMS, dims);

	if (sim_data.seq.analytical)

		md_copy(DIMS, dims, signals, x_magnetization, CFL_SIZE);

	else {

		complex float* tmp = md_alloc(DIMS, dims, CFL_SIZE);

		md_zsmul(DIMS, dims, tmp, y_magnetization, I);
		md_zadd(DIMS, dims, signals, x_magnetization, tmp);

		md_free(tmp);
	}

	unmap_cfl(DIMS, dims, signals);

	// Export z Component

	complex float* z_comp = NULL;

	if (NULL != z_component) {

		z_comp = create_cfl(z_component, DIMS, dims);

		md_copy(DIMS, dims, z_comp, z_magnetization, CFL_SIZE);

		unmap_cfl(DIMS, dims, z_comp);
	}

	// Determine radial component of magnetization

	complex float* radial_comp = NULL;

	if (NULL != radial_component) {

		radial_comp = create_cfl(radial_component, DIMS, dims);

		complex float* tmp = md_alloc(DIMS, dims, CFL_SIZE);
		complex float* tmp2 = md_alloc(DIMS, dims, CFL_SIZE);
		complex float* tmp3 = md_alloc(DIMS, dims, CFL_SIZE);

		md_zmul(DIMS, dims, tmp, x_magnetization, x_magnetization);
		md_zmul(DIMS, dims, tmp2, y_magnetization, y_magnetization);
		md_zmul(DIMS, dims, tmp3, z_magnetization, z_magnetization);

		md_zadd(DIMS, dims, tmp2, tmp2, tmp);
		md_zadd(DIMS, dims, tmp3, tmp3, tmp2);

		md_zsqrt(DIMS, dims, radial_comp, tmp3);

		md_free(tmp);
		md_free(tmp2);
		md_free(tmp3);

		unmap_cfl(DIMS, dims, radial_comp);
	}

	md_free(x_magnetization);
	md_free(y_magnetization);
	md_free(z_magnetization);

	return 0;
}


