
#include <complex.h>
#include <stdbool.h>


struct simdata_pulse {

	float pulse_length;
	float rf_start;
	float rf_end;
	float flipangle;
	float phase;
	float nl;		/*number of zero crossings to the left of the main loop*/
	float nr; 		/*number of zero crossings to the right of the main loop*/
	float n;		/*max(nl, nr)*/
	float t0;		/*time of main lope: t0 =  = pulse_len / ( 2 + (nl-1)  + (nr-1))*/
	float alpha; 		/*windows of pulse ( 0: normal sinc, 0.5: Hanning, 0.46: Hamming)*/
	float A;		/*offset*/
	float energy_scale;	/*Define energy scale factor*/
	bool pulse_applied;
};
extern const struct simdata_pulse simdata_pulse_defaults;


struct simdata_voxel {

	float r1;
	float r2;
	float m0;
	float w;
	bool spin_ensamble;
};
extern const struct simdata_voxel simdata_voxel_defaults;


struct simdata_seq {
	
	int analytical;
	int seq_type;
	float tr;
	float te;
	int rep_num;
	int spin_num;
	int num_average_rep;
	int run_num; /* Number of applied sequence trains*/
	
	complex float* slice_profile;
	complex float* variable_fa;
};
extern const struct simdata_seq simdata_seq_defaults;


struct simdata_tmp {

	float t;
	int rep_counter;
	int spin_counter;
	int run_counter;
};
extern const struct simdata_tmp simdata_tmp_defaults;


struct simdata_grad {

	float gb[3];
	float gb_eff[3];
};
extern const struct simdata_grad simdata_grad_defaults;


struct sim_data {

	struct simdata_seq seq;
	struct simdata_voxel voxel;
	struct simdata_pulse pulse;
	struct simdata_grad grad;
	struct simdata_tmp tmp;
	
};
// extern const struct sim_data simData_defaults;


struct bloch_s {

	float r1;
	float r2;
	float m0;
	float gb[3];
	float gb_eff[3];
};


void bloch_pdy2(void* _data, float* out, float t, const float* in);

void bloch_pdp2(void* _data, float* out, float t, const float* in);

void bloch_pdy3(void* _data, float* out, float t, const float* in);

void bloch_pdp3(void* _data, float* out, float t, const float* in);

void bloch_simu_fun2(void* _data, float* out, float t, const float* in);

void ADCcorr(int N, int P, float out[P + 2][N], float in[P + 2][N]);

void relaxation2(void* _data, float h, float tol, int N, int P, float xp[P + 2][N], float st, float end);

void isochrom_distribution( void* _data, float *isochromats );

void create_rf_pulse(void* _pulseData, float rf_start, float rf_end, float angle, float phase, float nl, float nr, float alpha);

void start_rf_pulse(void* _data, float h, float tol, int N, int P, float xp[P + 2][N]);

void ode_bloch_simulation3( void* _data, float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3]);

void create_sim_block(void* _data);

void run_sim_block(void* _data, float* mxy, float* sa_r1, float* sa_r2, float* saM0Signal, float h, float tol, int N, int P, float xp[P + 2][N], bool get_signal);
