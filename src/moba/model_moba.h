


#include "misc/mri.h"

struct linop_s;
struct nlop_s;
struct noir_model_conf_s;

struct moba_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
	const struct linop_s* linop_alpha;
};

struct irflash_conf_s {

	/*T1*/
	complex float* input_TI;

	/*MOLLI*/
	complex float* input_TI_t1relax;

	/*IR_phy_alpha_in*/
	complex float* input_alpha;
};
extern const struct irflash_conf_s irflash_conf_s_defaults;

struct bloch_conf_s {

	/*Simulation Parameter*/
	int sequence;
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

	/*Reconstruction Parameter*/
	float scale[4];
	float fov_reduction_factor;
	int rm_no_echo;
	bool full_ode_sim;
	int not_wav_maps;

	/*Input Calibrations*/
	complex float* input_b1;
	complex float* input_sliceprofile;
	complex float* input_fa_profile;
};
extern const struct bloch_conf_s bloch_conf_s_defaults;



typedef enum {IR, MOLLI, IR_SS, IR_phy, IR_phy_alpha_in, Bloch} moba_model;

struct moba_conf_s {

	/*All*/
	moba_model model;

	/*T1*/
	struct irflash_conf_s irflash_conf;

	/*Bloch*/
	struct bloch_conf_s bloch_conf;
};
// extern const struct moba_conf moba_conf_defaults;

extern struct moba_s moba_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf, const struct moba_conf_s* conf_model, _Bool usegpu);


