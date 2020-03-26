
#ifndef __RECON_MECO_H
#define __RECON_MECO_H

#include "noir/recon.h"

enum REGU {
	TIKHONOV,
	WAV,
	LLR,
};

extern void meco_recon(const struct noir_conf_s* conf, unsigned int sel_model, unsigned int sel_regu, bool out_origin_maps, const long maps_dims[DIMS], const long sens_dims[DIMS], _Complex float* x, _Complex float* xref, const _Complex float* pattern, const _Complex float* mask, const _Complex float* TE, const long ksp_dims[DIMS], const _Complex float* ksp);

#endif
