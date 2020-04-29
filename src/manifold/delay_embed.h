/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019. Sebastian Rosenzweig.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Sebastian Rosenzweig (sebastian.rosenzweig@med.uni-goettingen.de)
 */


#include "misc/mri.h"

struct delay_conf {
	
	long kernel_dims[3]; // Hankel kernel
	int window; 	 // window size (embedding dimension size)
	int normalize;	 // normalize data
	int rm_mean; 	 // remove mean of data
	_Bool zeropad; 	 // apply zeropadding
	float weight;	 // weigthing factor for delayed coordinates
	
	char* name_S; 	 // output name singular values
	char* backproj;  // output name backprojection
	long group;		 // bitmask for grouping
	int rank;	 	 // rank for backprojection

	// NLSA
	int nlsa_rank;	 // Smoothness of manifold

};

extern const struct delay_conf ssa_conf_default;
extern const struct delay_conf nlsa_conf_default;

extern bool check_selection(const long group, const int j);

extern void check_bp(struct delay_conf conf);

extern void preproc_ac(const long in_dims[DIMS], complex float* in, const struct delay_conf conf);

