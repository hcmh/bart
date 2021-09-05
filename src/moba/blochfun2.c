/* Author:
 * 2018-2021 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */


#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"

#include "nlops/nlop.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "model_Bloch.h"
#include "blochfun2.h"

#define round(x)	((int) ((x) + .5))



struct blochFun2_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* der_dims;
	const long* map_dims;
	const long* in_dims;
	const long* out_dims;

	const long* der_strs;
	const long* map_strs;
	const long* in_strs;
	const long* out_strs;

	float scale[5];

	//derivatives
	complex float* derivatives;
	complex float* tmp;
	complex float* in_tmp;

	complex float* input_sliceprofile;
	complex float* input_fa_profile;

	struct modBlochFit fitParameter;

	bool use_gpu;

	int counter;

	complex float* weights;
	const struct linop_s* linop_alpha;
};

DEF_TYPEID(blochFun2_s);

// FIXME: Join with T1_alpha.c function, but check weights!
static void moba_calc_weights(const long dims[3], complex float* dst)
{
	unsigned int flags = 0;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);


	klaplace(3, dims, dims, flags, dst);
	md_zsmul(3, dims, dst, dst, 440.);
	md_zsadd(3, dims, dst, dst, 1.);
	md_zspow(3, dims, dst, dst, -10.);
}

const struct linop_s* Bloch_get_alpha_trafo(struct nlop_s* op)
{
	const nlop_data_t* _data = nlop_get_data(op);
	struct blochFun2_s* data = CAST_DOWN(blochFun2_s, _data);
	return data->linop_alpha;
}

// FIXME: Join with T1_alpha.c function
void Bloch_forw_alpha(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

// FIXME: Join with T1_alpha.c function
void Bloch_back_alpha(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}


static void Bloch_fun2(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	double starttime = timestamp();
	debug_printf(DP_DEBUG2, "Started Forward Calculation\n");

	struct blochFun2_s* data = CAST_DOWN(blochFun2_s, _data);

	// Print out intermediate reconstruction steps

	if (DP_DEBUG2 <= debug_level) {

		// B1 map: kspace -> pixel

		long index[DIMS];
		md_set_dims(DIMS, index, 0);

		index[COEFF_DIM] = 3;

		const struct linop_s* linop_fftc = linop_fftc_create(DIMS, data->map_dims, FFT_FLAGS);

		md_clear(data->N, data->in_dims, data->in_tmp, CFL_SIZE);
		md_copy(data->N, data->in_dims, data->in_tmp, src, CFL_SIZE);

		md_copy_block(DIMS, index, data->map_dims, data->tmp, data->in_dims, data->in_tmp, CFL_SIZE);
		linop_forward_unchecked(linop_fftc, data->tmp, data->tmp);
		md_copy_block(DIMS, index, data->in_dims, data->in_tmp, data->map_dims, data->tmp, CFL_SIZE);

		linop_free(linop_fftc);

		// Dump cfl

		char name[255] = {'\0'};

		sprintf(name, "current_map_%02d", data->counter);
		dump_cfl(name, data->N, data->in_dims, data->in_tmp);

		data->counter++;
	}


	// Allocate GPU memory
	complex float* r1scale_tmp;
	complex float* r2scale_tmp;
	complex float* m0scale_tmp;
	complex float* b1scale_tmp;


#ifdef USE_CUDA
	if (data->use_gpu) {

		r1scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		r2scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		m0scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		b1scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
	}
	else {

		r1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		r2scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		m0scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		b1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	}
#else
	r1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	r2scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	m0scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	b1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
#endif


	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	//-------------------------------------------------------------------
	// Copy necessary files from GPU to CPU
	//-------------------------------------------------------------------

	// R1
	pos[COEFF_DIM] = 0;
	const complex float* R1 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, r1scale_tmp, data->map_strs, R1, (0. != data->scale[0]) ? data->scale[0] : 1.);

	complex float* r1scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, r1scale, r1scale_tmp, CFL_SIZE);

	// M0
	pos[COEFF_DIM] = 1;
	const complex float* M0 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, m0scale_tmp, data->map_strs, M0, (0. != data->scale[1]) ? data->scale[1] : 1.);

	complex float* m0scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, m0scale, m0scale_tmp, CFL_SIZE);

	// R2
	pos[COEFF_DIM] = 2;
	const complex float* R2 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, r2scale_tmp, data->map_strs, R2, (0. != data->scale[2]) ? data->scale[2] : 1.);

	complex float* r2scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, r2scale, r2scale_tmp, CFL_SIZE);

	// B1
	pos[COEFF_DIM] = 3;
	const complex float* B1 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp, data->map_strs, B1, (0. != data->scale[3]) ? data->scale[3] : 1.);

	Bloch_forw_alpha(data->linop_alpha, b1scale_tmp, data->tmp);	// freq -> pixel + smoothing!

	complex float* b1scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, b1scale, b1scale_tmp, CFL_SIZE);

	md_free(r1scale_tmp);
	md_free(r2scale_tmp);
	md_free(m0scale_tmp);
	md_free(b1scale_tmp);


	//Allocate Output CPU memory
	complex float* sig_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dr1_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dr2_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dm0_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);
	complex float* db1_cpu = md_alloc(data->N, data->out_dims, CFL_SIZE);


	long sliceprofile_dims[DIMS];

	complex float* sliceprofile_cpu = NULL;

	if (NULL != data->input_sliceprofile) {

		md_set_dims(DIMS, sliceprofile_dims, 1);
		sliceprofile_dims[READ_DIM] = data->fitParameter.sliceprofile_spins;

		sliceprofile_cpu = md_alloc(data->N, sliceprofile_dims, CFL_SIZE);
		md_copy(data->N, sliceprofile_dims, sliceprofile_cpu, data->input_sliceprofile, CFL_SIZE);

		debug_printf(DP_DEBUG2, "\n Slice Profile Estimates:\t");
		for (int i = 0; i < data->fitParameter.sliceprofile_spins; i++)
			debug_printf(DP_DEBUG2, "%f\t", cabsf(sliceprofile_cpu[i]));
		debug_printf(DP_DEBUG2, "\n");
	}

	long vfa_dims[DIMS];

	complex float* var_fa_cpu = NULL;

	if (NULL != data->input_fa_profile) {

		md_set_dims(DIMS, vfa_dims, 1);
		vfa_dims[READ_DIM] = data->fitParameter.num_vfa;

		var_fa_cpu = md_alloc(data->N, vfa_dims, CFL_SIZE);
		md_copy(data->N, vfa_dims, var_fa_cpu, data->input_fa_profile, CFL_SIZE);
	}

	//Prepare reduced FOV
	md_zfill(data->N, data->out_dims, sig_cpu, 0.);
	md_zfill(data->N, data->out_dims, dr1_cpu, 0.);
	md_zfill(data->N, data->out_dims, dr2_cpu, 0.);
	md_zfill(data->N, data->out_dims, dm0_cpu, 0.);
	md_zfill(data->N, data->out_dims, db1_cpu, 0.);

	//Get start and end values of reduced F0V
	int xstart = round(data->map_dims[0]/2. - (data->fitParameter.fov_reduction_factor * data->map_dims[0])/2.);
	int xend = round(xstart + data->fitParameter.fov_reduction_factor * data->map_dims[0]);
	int ystart = round(data->map_dims[1]/2. - (data->fitParameter.fov_reduction_factor * data->map_dims[1])/2.);
	int yend = round(ystart + data->fitParameter.fov_reduction_factor * data->map_dims[1]);
	int zstart = round(data->map_dims[2]/2. - (data->fitParameter.fov_reduction_factor * data->map_dims[2])/2.);
	int zend = round(zstart + data->fitParameter.fov_reduction_factor * data->map_dims[2]);

	//Solve rounding bug if fov_reduction_factor becomes to small, Change to round-up macro?!
	if (0 == zend)
		zend = 1;

	debug_printf(DP_DEBUG2, "Scaling: %f,\t%f,\t%f,\t%f,\t%f\n",	data->scale[0], data->scale[1], data->scale[2],
									data->scale[3], data->scale[4]);
	debug_printf(DP_DEBUG2, "seq: %d,\trfDuration: %f,\tTR:%f,\tTE:%f,\trep:%d,\tav:%d,\tFA:%f\n",
								data->fitParameter.sequence, data->fitParameter.rfduration, data->fitParameter.tr,
								data->fitParameter.te, data->out_dims[TE_DIM], data->fitParameter.averaged_spokes,
								data->fitParameter.fa);

	int rm_first_echo = data->fitParameter.rm_no_echo;

	#pragma omp parallel for collapse(3)
	for (int x = xstart; x < xend; x++)
		for (int y = ystart; y < yend; y++)
			for (int z = zstart; z < zend; z++) {

				//Calculate correct spatial position
				long spa_pos[DIMS];

				md_copy_dims(DIMS, spa_pos, data->map_dims);

				spa_pos[0] = x;
				spa_pos[1] = y;
				spa_pos[2] = z;

				long spa_ind = md_calc_offset(data->N, data->map_strs, spa_pos) / CFL_SIZE;

				//-------------------------------------------------------------------
				// Define simulation parameter
				//-------------------------------------------------------------------

				struct sim_data sim_data;

				// General
				sim_data.seq = simdata_seq_defaults;
				sim_data.seq.seq_type = data->fitParameter.sequence;
				sim_data.seq.tr = data->fitParameter.tr;
				sim_data.seq.te = data->fitParameter.te;

				if (4 == sim_data.seq.seq_type)
					sim_data.seq.rep_num = data->fitParameter.num_vfa;
				else
					sim_data.seq.rep_num = (data->out_dims[TE_DIM] + rm_first_echo) * data->fitParameter.averaged_spokes;

				sim_data.seq.spin_num = data->fitParameter.sliceprofile_spins;
				sim_data.seq.num_average_rep = data->fitParameter.averaged_spokes;
				sim_data.seq.run_num = data->fitParameter.runs;
				sim_data.seq.inversion_pulse_length = data->fitParameter.inversion_pulse_length;
				sim_data.seq.prep_pulse_length = data->fitParameter.prep_pulse_length;
				sim_data.seq.look_locker_assumptions = data->fitParameter.look_locker_assumptions;

				// Voxel
				sim_data.voxel = simdata_voxel_defaults;
				sim_data.voxel.r1 = crealf(r1scale[spa_ind]);
				sim_data.voxel.r2 = crealf(r2scale[spa_ind]);
				sim_data.voxel.m0 = m0scale[spa_ind];
				sim_data.voxel.b1 = (safe_isnanf(crealf(b1scale[spa_ind]))) ? 0. : crealf(b1scale[spa_ind]); //1.;
				sim_data.voxel.w = 0.;

				// Pulse
				sim_data.pulse = simdata_pulse_defaults;
				sim_data.pulse.flipangle = data->fitParameter.fa;
				sim_data.pulse.rf_end = data->fitParameter.rfduration;
				sim_data.pulse.bwtp = data->fitParameter.bwtp;
				sim_data.grad = simdata_grad_defaults;
				sim_data.tmp = simdata_tmp_defaults;


				if (NULL != data->input_sliceprofile) {

					sim_data.seq.slice_profile = md_alloc(DIMS, sliceprofile_dims, CFL_SIZE);
					md_copy(DIMS, sliceprofile_dims, sim_data.seq.slice_profile, sliceprofile_cpu, CFL_SIZE);
				}

				if (NULL != data->input_fa_profile) {

					sim_data.seq.variable_fa = md_alloc(DIMS, vfa_dims, CFL_SIZE);
					md_copy(DIMS, vfa_dims, sim_data.seq.variable_fa, var_fa_cpu, CFL_SIZE);
				}

				//-------------------------------------------------------------------
				// Run simulation
				//-------------------------------------------------------------------

				complex float mxy_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
				complex float sa_r1_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
				complex float sa_r2_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
				complex float sa_m0_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];
				complex float sa_b1_sig[sim_data.seq.rep_num / sim_data.seq.num_average_rep][3];

				if (data->fitParameter.full_ode_sim || NULL != data->input_fa_profile)	//variable flipangles are only included into ode simulation yet
					ode_bloch_simulation3(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig, sa_b1_sig);
				else
					matrix_bloch_simulation(&sim_data, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig, sa_b1_sig);

				//-------------------------------------------------------------------
				// Copy simulation output to storage on CPU
				//-------------------------------------------------------------------

				long curr_pos[DIMS];
				md_copy_dims(DIMS, curr_pos, spa_pos);

				long position = 0;

				for (int i = 0, j = 0; j < (sim_data.seq.rep_num - rm_first_echo) / sim_data.seq.num_average_rep; j++) {


					assert(i <= data->out_dims[TE_DIM]);

					curr_pos[TE_DIM] = i;
					position = md_calc_offset(data->N, data->out_strs, curr_pos) / CFL_SIZE;

					//Scaling: dB/dRi = dB/dRis * dRis/dRi
					float a = 1.;

					if (5 == sim_data.seq.seq_type && 0 != sim_data.pulse.flipangle)
						a = 1./(sinf(sim_data.pulse.flipangle * M_PI/180.) * expf(-sim_data.voxel.r2 * sim_data.seq.te));

					dr1_cpu[position] = a * data->scale[4] * data->scale[0] * (sa_r1_sig[j+rm_first_echo][1] + sa_r1_sig[j+rm_first_echo][0] * I);
					dm0_cpu[position] = a * data->scale[4] * data->scale[1] * (sa_m0_sig[j+rm_first_echo][1] + sa_m0_sig[j+rm_first_echo][0] * I);
					dr2_cpu[position] = a * data->scale[4] * data->scale[2] * (sa_r2_sig[j+rm_first_echo][1] + sa_r2_sig[j+rm_first_echo][0] * I);
					db1_cpu[position] = a * data->scale[4] * data->scale[3] * (sa_b1_sig[j+rm_first_echo][1] + sa_b1_sig[j+rm_first_echo][0] * I);
					sig_cpu[position] = a * data->scale[4] * (mxy_sig[j+rm_first_echo][1] + mxy_sig[j+rm_first_echo][0] * I);

					i++;
				}
			}

	md_free(r1scale);
	md_free(r2scale);
	md_free(m0scale);
	md_free(b1scale);

	debug_printf(DP_DEBUG3, "Copy data\n");

	//-------------------------------------------------------------------
	// Collect data of Signal (potentionally on GPU)
	//-------------------------------------------------------------------

	md_copy(data->N, data->out_dims, dst, sig_cpu, CFL_SIZE);

	md_free(sig_cpu);

	//-------------------------------------------------------------------
	// Collect data of derivatives in single arrray
	//-------------------------------------------------------------------

	md_clear(data->N, data->der_dims, data->derivatives, CFL_SIZE);

	md_set_dims(data->N, pos, 0);

	pos[COEFF_DIM] = 0; // R1
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dr1_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 1; // M0
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dm0_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 2; // R2
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dr2_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 3; // B1
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, db1_cpu, CFL_SIZE);


	md_free(dr1_cpu);
	md_free(dr2_cpu);
	md_free(dm0_cpu);
	md_free(db1_cpu);

	if (NULL != data->input_sliceprofile)
		md_free(sliceprofile_cpu);

	if (NULL != data->input_fa_profile)
		md_free(var_fa_cpu);

	double totaltime = timestamp() - starttime;
	debug_printf(DP_DEBUG2, "Time = %.2f s\n", totaltime);
}


static void Bloch_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	debug_printf(DP_DEBUG3, "Start Derivative\n");

	struct blochFun2_s* data = CAST_DOWN(blochFun2_s, _data);

	// Transform B1 map component from freq to pixel domain

	long pos[data->N];
	md_set_dims(data->N, pos, 0);
	pos[COEFF_DIM] = 3;

	md_clear(data->N, data->in_dims, data->in_tmp, CFL_SIZE);
	md_copy(data->N, data->in_dims, data->in_tmp, src, CFL_SIZE);

	md_copy_block(data->N, pos, data->map_dims, data->tmp, data->in_dims, data->in_tmp, CFL_SIZE);
	Bloch_forw_alpha(data->linop_alpha, data->tmp, data->tmp); // freq -> pixel + smoothing!
	md_copy_block(data->N, pos, data->in_dims, data->in_tmp, data->map_dims, data->tmp, CFL_SIZE);

	// Estimate derivative of operator
	// FIXME: replace by linop to use automatic gram matrix calculation
	md_clear(data->N, data->out_dims, dst, CFL_SIZE);
	md_ztenmul(data->N, data->out_dims, dst, data->der_dims, data->derivatives, data->in_dims, data->in_tmp);
}



static void Bloch_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	debug_printf(DP_DEBUG3, "Start Adjoint\n");

	struct blochFun2_s* data = CAST_DOWN(blochFun2_s, _data);

	// Estimate adjoint derivative of operator
	// FIXME: replace by linop to use automatic gram matrix calculation
	md_clear(data->N, data->in_dims, data->in_tmp, CFL_SIZE);
	md_zfmacc2(data->N, data->der_dims, data->in_strs, data->in_tmp, data->out_strs, src, data->der_strs, data->derivatives);

	// Transform B1 map component from pixel to freq domain

	long pos[data->N];
	md_set_dims(data->N, pos, 0);
	pos[COEFF_DIM] = 3;

	md_copy_block(data->N, pos, data->map_dims, data->tmp, data->in_dims, data->in_tmp, CFL_SIZE);
	Bloch_back_alpha(data->linop_alpha, data->tmp, data->tmp); // pixel -> freq + smoothing!
	md_copy_block(data->N, pos, data->in_dims, data->in_tmp, data->map_dims, data->tmp, CFL_SIZE);

	md_clear(data->N, data->in_dims, dst, CFL_SIZE);
	md_copy(data->N, data->in_dims, dst, data->in_tmp, CFL_SIZE);
}


static void Bloch_del2(const nlop_data_t* _data)
{
	struct blochFun2_s* data = CAST_DOWN(blochFun2_s, _data);

	md_free(data->derivatives);
	md_free(data->tmp);
	md_free(data->in_tmp);
	md_free(data->weights);

	md_free(data->input_sliceprofile);
	md_free(data->input_fa_profile);

	xfree(data->der_dims);
	xfree(data->map_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->der_strs);
	xfree(data->map_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	linop_free(data->linop_alpha);

	xfree(data);
}


struct nlop_s* nlop_Bloch_create2(int N, const long der_dims[N], const long map_dims[N], const long out_dims[N], const long in_dims[N], const struct modBlochFit* fit_para, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct blochFun2_s, data);
	SET_TYPEID(blochFun2_s, data);

	PTR_ALLOC(long[N], derdims);
	md_copy_dims(N, *derdims, der_dims);
	data->der_dims = *PTR_PASS(derdims);

	PTR_ALLOC(long[N], allstr);
	md_calc_strides(N, *allstr, der_dims, CFL_SIZE);
	data->der_strs = *PTR_PASS(allstr);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nmstr);

	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(long[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
	data->in_strs = *PTR_PASS(nistr);


	data->N = N;
	data->scale[0] = fit_para->scale[0];	// dR1 scaling
	data->scale[1] = fit_para->scale[1];	// dM0 scaling
	data->scale[2] = fit_para->scale[2];	// dR2 scaling
	data->scale[3] = fit_para->scale[3];	// dB1 scaling
	data->scale[4] = fit_para->scale[4];	// signal scaling

	data->derivatives = my_alloc(N, der_dims, CFL_SIZE);
	data->tmp = my_alloc(N, map_dims, CFL_SIZE);
	data->in_tmp = my_alloc(N, in_dims, CFL_SIZE);

	if (NULL != fit_para->input_sliceprofile) {

		long sliceprofile_dims[DIMS];

		md_set_dims(DIMS, sliceprofile_dims, 1);
		sliceprofile_dims[READ_DIM] = fit_para->sliceprofile_spins;

		data->input_sliceprofile = md_alloc(N, sliceprofile_dims, CFL_SIZE);	// Works just on CPU yet

		md_copy(N, sliceprofile_dims, data->input_sliceprofile, fit_para->input_sliceprofile, CFL_SIZE);
	}
	else
		data->input_sliceprofile = NULL;


	if (NULL != fit_para->input_fa_profile) {

		long vfa_dims[DIMS];

		md_set_dims(DIMS, vfa_dims, 1);
		vfa_dims[READ_DIM] = fit_para->num_vfa;

		data->input_fa_profile = my_alloc(N, vfa_dims, CFL_SIZE);

		md_copy(N, vfa_dims, data->input_fa_profile, fit_para->input_fa_profile, CFL_SIZE);
	}
	else
		data->input_fa_profile = NULL;

	// Fitting parameter

	data->fitParameter.sequence = fit_para->sequence;
	data->fitParameter.rfduration = fit_para->rfduration;
	data->fitParameter.bwtp = fit_para->bwtp;
	data->fitParameter.tr = fit_para->tr;
	data->fitParameter.te = fit_para->te;
	data->fitParameter.fa = fit_para->fa;
	data->fitParameter.runs = fit_para->runs;

	data->fitParameter.sequence = fit_para->sequence;
	data->fitParameter.averaged_spokes = fit_para->averaged_spokes;
	data->fitParameter.fov_reduction_factor = fit_para->fov_reduction_factor;
	data->fitParameter.sliceprofile_spins = fit_para->sliceprofile_spins;
	data->fitParameter.num_vfa = fit_para->num_vfa;
	data->fitParameter.rm_no_echo = fit_para->rm_no_echo;
	data->fitParameter.full_ode_sim = fit_para->full_ode_sim;
	data->fitParameter.inversion_pulse_length = fit_para->inversion_pulse_length;
	data->fitParameter.prep_pulse_length = fit_para->prep_pulse_length;
	data->fitParameter.look_locker_assumptions = fit_para->look_locker_assumptions;
	data->use_gpu = use_gpu;

	data->counter = 0;

	// Smoothness penalty for alpha map: Sobolev norm

	long w_dims[N];
	md_select_dims(N, FFT_FLAGS, w_dims, map_dims);

	data->weights = md_alloc(N, w_dims, CFL_SIZE);
	moba_calc_weights(w_dims, data->weights);

	const struct linop_s* linop_wghts = linop_cdiag_create(N, map_dims, FFT_FLAGS, data->weights);
	const struct linop_s* linop_ifftc = linop_ifftc_create(N, map_dims, FFT_FLAGS);

	data->linop_alpha = linop_chain(linop_wghts, linop_ifftc);

	linop_free(linop_wghts);
	linop_free(linop_ifftc);

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), Bloch_fun2, Bloch_der2, Bloch_adj2, NULL, NULL, Bloch_del2);
}