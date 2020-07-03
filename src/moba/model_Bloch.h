

#ifndef MODEL_Bloch_H
#define MODEL_Bloch_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

struct linop_s;
struct nlop_s;
struct noir_model_conf_s;

struct modBloch_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
};


struct modBlochFit {
	
	/*Simulation Parameter*/
	int sequence;
	float rfduration;
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

extern const struct modBlochFit modBlochFit_defaults;

extern struct modBloch_s bloch_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf, const struct modBlochFit* fit_para, _Bool usegpu);


#include "misc/cppwrap.h"


#endif
