#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <memory.h>
#include <complex.h>
#include <sys/time.h>


#include "num/ode.h"
#include "simu/bloch.h"
#include "misc/opts.h"
#include "num/init.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "simu/phantom.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/seq_model.h"
#include "misc/debug.h"
#include "num/fft.h"


//static const char usage_str[] = "<info-filename> <save-filename [*.txt]>";
static const char usage_str[] = "<out:signal> <out:sensT1> <out:sensT2> <out:sensDens>";
static const char help_str[] =
		"Creating simulated phantom and sens. maps.\n";


__attribute__((optimize("-fno-finite-math-only")))
int main_bloch(int argc, char* argv[argc])
{   
	
	//Measure time elapse
	struct timeval t_start, t_end;
	double elapsedTime;
	gettimeofday(&t_start, NULL);
	
	
	
	int seq = 0;
	int xdim = 128;
	int ydim = 128;
	float tr = 0.0045;
	float te = 0.00225;
	float flipangle = 45.;
	float rf_end = 0.0009;
	
	int aver_num = 1;
	int spin_num = 1;
	int repetition = 500;
	float offresonance = 0.;
	int runs = 1;
	
	const char* inputRel1 = NULL;
	const char* inputRel2 = NULL;
	const char* inputM0 = NULL;

	float m0i = 1;
	float t1i = WATER_T1;
	float t2i = WATER_T2;

	bool analytical = false;
	bool spin_ensamble = false;
	int kspace = 0;
	bool linear_offset = false;
	bool operator_sim = false;
	const char* fa_file = NULL;
	const char* spherical_coord = NULL;
	
	const struct opt_s opts[] = {

		/* Sequence Info */
		OPT_INT('s', &seq, "sequence", "options: 0 = bSSFP[default], 1 = invbSSFP, 3 = pcbSSFP, 4 = inv. bSSFP without preparation, 5 = invFLASH, 6 = invpcbSSFP"),
		OPT_INT('x', &xdim, "n", "dimensions in x for shepp-logan phantom"),
		OPT_INT('y', &ydim, "n", "dimensions in y for shepp-logan phantom"),
		OPT_FLOAT('t', &tr, "", "TR [s]"),
		OPT_FLOAT('e', &te, "", "TE [s]"),
		OPT_FLOAT('f', &flipangle, "", "Flip angle [deg]"),
		OPT_FLOAT('p', &rf_end, "", "RF pulse duration [s]"),

		/* Voxel Info */
		OPT_INT('a', &aver_num, "n", "number of averaged TRs"),
		OPT_INT('n', &spin_num, "n", "number of spins"),
		OPT_INT('r', &repetition, "n", "repetitions/train-length"),
		OPT_FLOAT('w', &offresonance, "", "off-resonance frequency [rad]"),
		OPT_INT('X', &runs, "", "runs of sequence"),

		/* Input Maps */
		OPT_STRING('I', &inputRel1, "Input Rel1", "Input relaxation parameter 1."),
		OPT_STRING('i', &inputRel2, "Input Rel2", "Input relaxation parameter 2."),
		OPT_STRING('M', &inputM0, "Input M0", "Input M0."),

		/*for x == 1 && y == 1 */
		OPT_FLOAT('m', &m0i, "M0 [s]", "for x & y == 1"),
		OPT_FLOAT('1', &t1i, "T1 [s]", "for x & y == 1"),
		OPT_FLOAT('2', &t2i, "T2 [s]", "for x & y == 1"),

		/* Special Cases */
		OPT_SET('A', &analytical, "Use analytical model for simulation"),
		OPT_INT('d', &debug_level, "level", "Debug level"),   
		OPT_SET('E', &spin_ensamble, "Spin Ensample"),
		OPT_INT('k', &kspace, "d", "kspace output? default:0=no"),
		OPT_SET('L', &linear_offset, "Add linear distribution of off-set freq."),
		OPT_SET('O', &operator_sim, "Simulate using operator based simulation."),
		OPT_STRING('F', &fa_file, "", "Variable flipangle file"),
		OPT_STRING('c', &spherical_coord, "", "Output spherical coordinates: r "),
	};
	
	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);
	num_init();
	
	/* --------------------------------------------------------------
	* ---------------  Starting Simulation  ------------------------
	* --------------------------------------------------------------
		Sequences:
		(seq) {
			case 0: 	normal bSSFP
			case 1: 	inversion recovery prepared bSSFP
			case 2:		FLASH
			case 3: 	pcbSSFP
			case 5: 	inversion recovery prepared FLASH
			case 6: 	inversion recovery prepared pcbSSFP
			
		}
	* --------------------------------------------------------------
	*/
	
	// Check cases
	if (operator_sim && (seq == 3 || seq == 6))
		error( "Simulation tool does not allow to simulate pcbSSFP sequences using matrix exponentials yet.\n" );
	
	if (operator_sim && (rf_end == 0.))
		error( "Simulation tool does not allow to hard-pulses using matrix exponentials yet.\n" );
	
	if (operator_sim && (NULL != fa_file))
		error( "Simulation tool does not allow variable flipangles for operator based simulation.\n" );
	
	
	long dim_map[DIMS] = { [0 ... DIMS - 1] = 1 };

	complex float* map_T1;
	complex float* map_T2;
	complex float* map_M0;
	
	if (xdim != 1 && ydim != 1) {
		
		// Create Shepp-Logan phantom if no maps are given
		if ((NULL == inputRel1) && (NULL == inputRel2) && (NULL == inputM0)) {
			
			dim_map[0] = xdim;
			dim_map[1] = ydim;
			
			map_T1 = md_alloc(DIMS, dim_map, CFL_SIZE);
			map_T2 = md_alloc(DIMS, dim_map, CFL_SIZE);
			map_M0 = md_alloc(DIMS, dim_map, CFL_SIZE);
			
			complex float* samples = NULL;
			long sstrs[DIMS];		
			
			complex float* map = create_cfl("T1T2Parameter", DIMS, dim_map);
			md_clear(DIMS, dim_map, map, CFL_SIZE);
			calc_phantom_t1t2(NULL, dim_map, map, false/*kspace*/, sstrs, samples);
			
			complex float* dens = create_cfl("DensParameter", DIMS, dim_map);
			md_zfill(DIMS, dim_map, dens, 1.0);
			
			md_zreal(DIMS, dim_map, map_T1, map);
			md_zimag(DIMS, dim_map, map_T2, map);
			md_zabs(DIMS, dim_map, map_M0, dens);
			
			unmap_cfl(DIMS, dim_map, map);
			unmap_cfl(DIMS, dim_map, dens);
		} 
		else { // Create Phantom based on given Relaxation and M0 Maps
			
			complex float* input_map_T1 = NULL;
			
			if (NULL != inputRel1) 
				input_map_T1 = load_cfl(inputRel1, DIMS, dim_map);
			else
				debug_printf(DP_WARN, "No input for relaxation parameter 1 could be found!");
			
			complex float* input_map_T2 = NULL;
			
			if (NULL != inputRel2) 
				input_map_T2 = load_cfl(inputRel2, DIMS, dim_map);
			else
				debug_printf(DP_WARN, "No input for relaxation parameter 2 could be found!");
				
			complex float* input_map_M0 = NULL;
			
			if (NULL != inputM0) 
				input_map_M0 = load_cfl(inputM0, DIMS, dim_map);
			else
				debug_printf(DP_WARN, "No input for M0 could be found!");
			
			
			map_T1 = md_alloc(DIMS, dim_map, CFL_SIZE);
			map_T2 = md_alloc(DIMS, dim_map, CFL_SIZE);
			map_M0 = md_alloc(DIMS, dim_map, CFL_SIZE);
			
			md_copy(DIMS, dim_map, map_T1, input_map_T1, CFL_SIZE);
			md_copy(DIMS, dim_map, map_T2, input_map_T2, CFL_SIZE);
			md_copy(DIMS, dim_map, map_M0, input_map_M0, CFL_SIZE);
			
			xdim = dim_map[0];
			ydim = dim_map[1];
			
			if (NULL != inputRel1)
				unmap_cfl(DIMS, dim_map, input_map_T1);
			
			if (NULL != inputRel2)
				unmap_cfl(DIMS, dim_map, input_map_T2);
			
			if (NULL != inputM0)
				unmap_cfl(DIMS, dim_map, input_map_M0);
		}
	}
	else {
		
		dim_map[0] = xdim;
		dim_map[1] = ydim;
		map_T1 = md_alloc(DIMS, dim_map, CFL_SIZE);
		map_T2 = md_alloc(DIMS, dim_map, CFL_SIZE);
		map_M0 = md_alloc(DIMS, dim_map, CFL_SIZE);
		
	}

	long dim_phantom[DIMS] = { [0 ... DIMS - 1] = 1 };
	
	dim_phantom[0] = dim_map[0];
	dim_phantom[1] = dim_map[1];
	dim_phantom[TE_DIM] = repetition / aver_num ;

	complex float* phantom = create_cfl(argv[1], DIMS, dim_phantom);
	complex float* sensitivitiesT1 = create_cfl(argv[2], DIMS, dim_phantom);
	complex float* sensitivitiesT2 = create_cfl(argv[3], DIMS, dim_phantom);
	complex float* sensitivitiesDens = create_cfl(argv[4], DIMS, dim_phantom);
	complex float* r_out = NULL;
	
	if (NULL != spherical_coord)
		r_out = create_cfl(spherical_coord, DIMS, dim_phantom);
	

	long dim_vfa[DIMS] = { [0 ... DIMS - 1] = 1 };
	
	complex float* vfa_file = NULL;
	
	if (NULL != fa_file)
		vfa_file = load_cfl(fa_file, DIMS, dim_vfa);
	
	
	struct HSFP_model hsfp_data2 = hsfp_defaults;
	if ( 4 == seq && analytical) {
		
		hsfp_data2.tr = tr;
		hsfp_data2.repetitions = repetition;
		hsfp_data2.beta = -1;
		hsfp_data2.pa_profile = md_alloc(DIMS, dim_vfa, CFL_SIZE);
		md_copy(DIMS, dim_vfa, hsfp_data2.pa_profile, vfa_file, CFL_SIZE);
	} 
	
	struct LookLocker_model ll_data = looklocker_defaults;
	if ( 5 == seq && analytical) {
		ll_data.tr = tr;
		ll_data.repetitions = repetition;
	}
	
	struct IRbSSFP_model irbSSFP_data = IRbSSFP_defaults;
	if ( 1 == seq && analytical) {
		irbSSFP_data.tr = tr;
		irbSSFP_data.repetitions = repetition;
	}
		
	
	#pragma omp parallel for collapse(2)
	for (int x = 0; x < dim_phantom[0]; x++) 
		for (int y = 0; y < dim_phantom[1]; y++) {

			float t1;
			float t2;
			float m0;
			
			if (xdim != 1 && ydim != 1) {

				t1 = crealf(map_T1[(y * dim_phantom[0]) + x]);  
				t2 = cimagf(map_T2[(y * dim_phantom[0]) + x]); 
				m0 = cabsf(map_M0[(y * dim_phantom[0]) + x]);
			}
			else {

				t1 = t1i;
				t2 = t2i;
				m0 = m0i;
			}
			
			//Skip empty voxels
			if (t1 <= 0.001 || t2 <= 0.001) { 
				
				for (int z = 0; z < dim_phantom[TE_DIM]; z++) {
					
					phantom[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
					sensitivitiesT1[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
					sensitivitiesT2[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
					sensitivitiesDens[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
				}
					
				continue;
			}

			struct SimData sim_data;
			
			sim_data.seqData = seqData_defaults;
			sim_data.seqData.seq_type = seq;
			sim_data.seqData.TR = tr;
			sim_data.seqData.TE = te;
			
			if (NULL != vfa_file) {
				
				sim_data.seqData.variable_fa = md_alloc(DIMS, dim_vfa, CFL_SIZE);
				md_copy(DIMS, dim_vfa, sim_data.seqData.variable_fa, vfa_file, CFL_SIZE);
				
				sim_data.seqData.rep_num = dim_vfa[0];
			}
			else
				sim_data.seqData.rep_num = repetition;
			
			sim_data.seqData.spin_num = spin_num;
			sim_data.seqData.num_average_rep = aver_num;
			sim_data.seqData.run_num = runs;
			
			if (NULL != vfa_file) {
				
				sim_data.seqData.variable_fa = md_alloc(DIMS, dim_vfa, CFL_SIZE);
				md_copy(DIMS, dim_vfa, sim_data.seqData.variable_fa, vfa_file, CFL_SIZE);
			}
			
			sim_data.voxelData = voxelData_defaults;
			sim_data.voxelData.r1 = 1 / t1;
			sim_data.voxelData.r2 = 1 / t2;
			sim_data.voxelData.m0 = m0;
			sim_data.voxelData.spin_ensamble = spin_ensamble;
			
			if (linear_offset)
				sim_data.voxelData.w = (float) y / (float) dim_phantom[1]  * M_PI / sim_data.seqData.TE; //Get offset values from -pi to +pi
			else
				sim_data.voxelData.w = offresonance;
			
			sim_data.pulseData = pulseData_defaults;
			sim_data.pulseData.flipangle = flipangle;
			sim_data.pulseData.RF_end = rf_end;
			sim_data.gradData = gradData_defaults;
			sim_data.seqtmp = seqTmpData_defaults;
			
			float mxySig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saR1Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saR2Sig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			float saDensSig[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep][3];
			
			float signal[sim_data.seqData.rep_num / sim_data.seqData.num_average_rep];	// radial magnetization

			float fa = flipangle * M_PI / 180.; //conversion to rad
			
			if (analytical) {
				
				if( 4 == seq && NULL != spherical_coord) {
					
					struct HSFP_model hsfp_data = hsfp_data2;
					
					hsfp_data.t1 = t1;
					hsfp_data.t2 = t2;
					
					hsfp_simu(&hsfp_data, signal);
					
					int ind = 0;
					
					for (int z = 0; z < dim_phantom[TE_DIM]; z++) {
						
						ind = (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x;
						
						phantom[ind] = sinf(cabsf(vfa_file[z])) * signal[z];
						sensitivitiesT1[ind] = 0.;
						sensitivitiesT2[ind] = 0.;
						sensitivitiesDens[ind] = 0.;
						
						r_out[ind] = fabsf(signal[z]);
					}
				} else if (5 == seq) {
					
					struct LookLocker_model ll_data2 = ll_data;
					
					ll_data2.t1 = t1;
					ll_data2.m0 = m0;
					ll_data2.fa = fa;
					
					looklocker_simu(&ll_data2, signal);
					
					for (int z = 0; z < dim_phantom[TE_DIM]; z++) {
						
						phantom[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = signal[z];
						sensitivitiesT1[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
						sensitivitiesT2[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
						sensitivitiesDens[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
					}
				} else {
					struct IRbSSFP_model irbSSFP_data2 = irbSSFP_data;
					
					irbSSFP_data2.t1 = t1;
					irbSSFP_data2.t2 = t2;
					irbSSFP_data2.m0 = m0;
					irbSSFP_data2.fa = fa;
					
					IR_bSSFP_simu(&irbSSFP_data2, signal);
					
					for (int z = 0; z < dim_phantom[TE_DIM]; z++) {
						
						phantom[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = signal[z];
						sensitivitiesT1[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
						sensitivitiesT2[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
						sensitivitiesDens[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = 0.;
					}
				}
			}
			else {	//start ODE based simulation

				if (operator_sim)
					matrix_bloch_simulation(&sim_data, mxySig, saR1Sig, saR2Sig, saDensSig);
				else
					ode_bloch_simulation3(&sim_data, mxySig, saR1Sig, saR2Sig, saDensSig);


				//Add data to phantom
				for (int z = 0; z < dim_phantom[TE_DIM]; z++) {
					
					//changed x-and y-axis to have same orientation as measurements
					sensitivitiesT1[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = saR1Sig[z][1] + saR1Sig[z][0] * I; 
					sensitivitiesT2[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = saR2Sig[z][1] + saR2Sig[z][0] * I;
					sensitivitiesDens[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = saDensSig[z][1] + saDensSig[z][0] * I;
					phantom[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = mxySig[z][1] + mxySig[z][0] * I;
					
					if (NULL != spherical_coord)
						r_out[ (z * dim_phantom[0] * dim_phantom[1]) + (y * dim_phantom[0]) + x] = sqrtf(mxySig[z][0] * mxySig[z][0] + 
																mxySig[z][1] * mxySig[z][1] + 
																mxySig[z][2] * mxySig[z][2]);
				}
			}
		}

	if (kspace) {

		complex float* ksp = create_cfl("phantom_ksp", DIMS, dim_phantom);

		fftuc(DIMS, dim_phantom, FFT_FLAGS, ksp, phantom);

		unmap_cfl(DIMS, dim_phantom, ksp);
	}

	if (1 != xdim && 1 != ydim) {

		md_free(map_T1);
		md_free(map_T2);
		md_free(map_M0);
	}

	unmap_cfl(DIMS, dim_phantom, phantom);    
	unmap_cfl(DIMS, dim_phantom, sensitivitiesT1);
	unmap_cfl(DIMS, dim_phantom, sensitivitiesT2);
	unmap_cfl(DIMS, dim_phantom, sensitivitiesDens);
	
	if (NULL != spherical_coord)
		unmap_cfl(DIMS, dim_phantom, r_out);

	//Calculate and print elapsed time 
	gettimeofday(&t_end, NULL);
	elapsedTime = ( t_end.tv_sec - t_start.tv_sec ) * 1000 + ( t_end.tv_usec - t_start.tv_usec ) / 1000;
	int min = (int) elapsedTime / 1000 / 60;
	int sec = ( ( (int) elapsedTime - min * 60 * 1000 ) / 1000) % 60;
	int msec =  (int) elapsedTime - min * 60 * 1000 - sec * 1000;
	printf("Time: %d:%d:%d [min:s:ms]\n", min, sec, msec);

	return 0;
}
