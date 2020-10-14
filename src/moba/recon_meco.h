/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#ifndef __RECON_MECO_H
#define __RECON_MECO_H

#include "iter/iter.h"
#include "iter/iter2.h"

#include "noir/recon.h"

struct moba_conf;

void meco_recon(const struct moba_conf* moba_conf, 
		unsigned int sel_model, unsigned int sel_irgnm, bool real_pd, 
		unsigned int wgh_fB0, float scale_fB0, bool warmstart, iter_conf* iconf, 
		bool out_origin_maps, double scaling_Y, 
		const long maps_dims[DIMS], complex float* maps, 
		const long sens_dims[DIMS], complex float* sens, 
		complex float* x, complex float* xref, 
		const complex float* pattern, 
		const complex float* mask, 
		const complex float* TE, 
		const long ksp_dims[DIMS], 
		const complex float* ksp, 
		bool use_lsqr);

#endif
