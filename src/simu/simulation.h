
#include <complex.h>
#include <stdbool.h>


struct PulseData {

	float pulse_length;
	float RF_start;
	float RF_end;
	float flipangle;
	float phase;
	float nl;						/*number of zero crossings to the left of the main loop*/
	float nr; 						/*number of zero crossings to the right of the main loop*/
	float n;						/*max(nl, nr)*/
	float t0;						/*time of main lope: t0 =  = pulse_len / ( 2 + (nl-1)  + (nr-1))*/
	float alpha; 					/*windows of pulse ( 0: normal sinc, 0.5: Hanning, 0.46: Hamming)*/
	float A;						/*offset*/
	float energy_scale;				/*Define energy scale factor*/
	bool pulse_applied;
};
extern const struct PulseData pulseData_defaults;


struct VoxelData {

	float r1;
	float r2;
	float m0;
	float w;
	bool spin_ensamble;
};
extern const struct VoxelData voxelData_defaults;


struct SeqData {

	int seq_type;
	float TR;
	float TE;
	int rep_num;
	int spin_num;
	int num_average_rep;
	
	complex float* variable_fa;
};
extern const struct SeqData seqData_defaults;


struct SeqTmpData {

	float t;
	int rep_counter;
	int spin_counter;
};
extern const struct SeqTmpData seqTmpData_defaults;


struct GradData {

	float gb[3];
	float gb_eff[3];
};
extern const struct GradData gradData_defaults;



struct SimData {

	struct SeqData seqData;
	struct VoxelData voxelData;
	struct PulseData pulseData;
	struct GradData gradData;
	struct SeqTmpData seqtmp;
	
};
// extern const struct SimData simData_defaults;


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

void isochromDistribution( void* _data, float *isochromats );

void create_rf_pulse(void* _pulseData, float RF_start, float RF_end, float angle, float phase, float nl, float nr, float alpha);

void start_rf_pulse(void* _data, float h, float tol, int N, int P, float xp[P + 2][N]);

void ode_bloch_simulation3( void* _data, float (*mxyOriSig)[3], float (*saT1OriSig)[3], float (*saT2OriSig)[3], float (*densOriSig)[3], complex float* input_sp);

void create_sim_block(void* _data);

void run_sim_block(void* _data, float* mxySignal, float* saR1Signal, float* saR2Signal, float* saM0Signal, float h, float tol, int N, int P, float xp[P + 2][N], bool get_signal);
