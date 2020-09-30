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
	bool EOF_info;	 // in-phase information

	bool temporal_nn; 	 // temporal nearest neighbors
	float lambda_nn; 	// lambda_nn
	
	// NLSA
	_Bool nlsa;	     	// is nlsa
	long nlsa_rank;	 	// smoothness of manifold
	char* name_tbasis;  // output name of temporal basis for NLSA
	_Bool L_out;		// output Laplacian
	_Bool basis_out;	// output Laplace-Beltrami basis
};

extern const struct delay_conf ssa_conf_default;
extern const struct delay_conf nlsa_conf_default;

extern bool check_selection(const long group, const int j);

extern void check_bp(struct delay_conf* conf);

extern void preproc_ac(const long in_dims[DIMS], complex float* in, const struct delay_conf conf);

extern void weight_delay(const long A_dims[2], complex float* A, const struct delay_conf conf);


