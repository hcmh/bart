
#include <complex.h>
#include <stdbool.h>


#include "simu/pulse.h"

#ifndef SIMULATION_H
#define SIMULATION_H

struct simdata_voxel {

	float r1;
	float r2;
	complex float m0;
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
	float inversion_pulse_length;
	float prep_pulse_length;
	int molli_break;
	int molli_measure;
	bool look_locker_assumptions;
	
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


extern void relaxation2(struct sim_data* data, float h, float tol, int N, int P, float xp[P + 1][N], float st, float end);
extern void isochrom_distribution(struct sim_data* data, float* isochromats);
extern void start_rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P + 1][N]);
extern void ode_bloch_simulation3(struct sim_data* data, complex float (*mxy_sig)[3], complex float (*sa_r1_sig)[3], complex float (*sa_r2_sig)[3], complex float (*sa_m0_sig)[3], complex float (*sa_b1_sig)[3]);
extern void create_sim_block(struct sim_data* data);

extern void bloch_simulation(unsigned int D, struct sim_data* sim_data, const long dims[D], complex float* m, complex float* d, bool ode);
extern void bloch_simulation2(unsigned int D, struct sim_data* sim_data, const long dims[D], complex float* m, complex float* d, bool ode, int custom_dim);
extern void bloch_simulation_mag(unsigned int D, struct sim_data* sim_data, const long dims[D], complex float* m, bool ode);
extern void bloch_xy_simulation(unsigned int D, struct sim_data* sim_data, const long dims[D], complex float* sig, complex float* der, bool ode);

extern void bloch_simulation_crb(int N, int P, struct sim_data* sim_data, float crb[P], bool ode);

#endif