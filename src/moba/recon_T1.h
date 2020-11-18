
#ifndef RECON_T1_H
#define RECON_T1_H

#include "misc/mri.h"

#include "grecon/optreg.h"

struct moba_conf {

	unsigned int iter;
	unsigned int opt_reg;
	float alpha;
	float alpha_min;
	float redu;
	float step;
	float lower_bound;
	float tolerance;
	unsigned int inner_iter;
	bool noncartesian;
	bool auto_norm_off;
	bool sms;
	bool MOLLI;
	bool k_filter;
	bool IR_SS;
	float IR_phy;
	int algo;
	float rho;
	struct opt_reg_s ropts;
};



extern struct moba_conf moba_defaults;


extern void T1_recon(const struct moba_conf* conf, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* pattern, const _Complex float* mask, const _Complex float* TI, const _Complex float* TI_t1relax, const _Complex float* kspace_data, _Bool usegpu);


#endif
