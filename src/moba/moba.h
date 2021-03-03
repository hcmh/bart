
#include <stdbool.h>
#include <complex.h>

struct opt_reg_s;


struct irflash_conf_s {

	/*T1*/
	complex float* input_TI;

	/*MOLLI*/
	complex float* input_TI_t1relax;

	/*IR_phy_alpha_in*/
	complex float* input_alpha;
};
extern const struct irflash_conf_s irflash_conf_s_defaults;

typedef enum sim_seq_t {bSSFP, IRbSSFP, FLASH, pcbSSFP, IRbSSFP_wo_prep, IRFLASH, IRpcbSSFP, NONE} moba_sim_seq;
typedef enum sim_type_t {OBS, ODE} moba_sim_type;

struct sim_conf_s {

	/*Simulation Parameter*/
	moba_sim_seq sequence;
	float rfduration;
	float bwtp;
	float tr;
	float te;
	int averaged_spokes;
	int sliceprofile_spins;
	int num_vfa;
	float fa;
	int runs; /*Number of applied sequence trains*/
	float inversion_pulse_length;
	float prep_pulse_length;
	bool look_locker_assumptions;

	/*Reconstruction Parameter*/
	float scale[4];
	float fov_reduction_factor;
	int rm_no_echo;
	moba_sim_type sim_type;
	int not_wav_maps;

	/*Input Calibrations*/
	complex float* input_b1;
	complex float* input_sliceprofile;
	complex float* input_fa_profile;
};
extern const struct sim_conf_s sim_conf_s_defaults;


struct opt_conf_s {

	unsigned int iter;
	unsigned int opt_reg;
	float alpha;
	float alpha_min;
	float redu;
	float step;
	float lower_bound;
	float tolerance;
	float damping;
	unsigned int inner_iter;
	bool noncartesian;
        bool sms;
	bool k_filter;
	bool auto_norm_off;
	bool stack_frames;
	int algo;	// enum algo_t
	float rho;
	struct opt_reg_s* ropts;
};

extern struct opt_conf_s opt_conf_s_defaults;


// FIXME: Remove and unify all moba function to support moba_conf_s
struct moba_conf {

	unsigned int iter;
	unsigned int opt_reg;
	float alpha;
	float alpha_min;
	float redu;
	float step;
	float lower_bound;
	float tolerance;
	float damping;
	unsigned int inner_iter;
	bool noncartesian;
        bool sms;
	bool k_filter;
	bool MOLLI;
	bool IR_SS;
	float IR_phy;
	bool auto_norm_off;
	bool stack_frames;
	int algo;	// enum algo_t
	float rho;
	struct opt_reg_s* ropts;

	complex float* input_alpha;
};

extern struct moba_conf moba_defaults;



typedef enum moba_t {IR, MOLLI, IR_SS, IR_phy, IR_phy_alpha_in, T2, MGRE, Bloch} moba_model;

struct moba_conf_s {

	/*All*/
	moba_model model;

	/*T1, FIXME: Make most unnecessary using simulation sim_conf_s*/
	struct irflash_conf_s irflash;

	/*Bloch*/
	struct sim_conf_s sim;

	/*Optimization: FIXME: Replace by opt_conf_s*/
	struct moba_conf opt;
};
// extern const struct moba_conf moba_conf_defaults;


