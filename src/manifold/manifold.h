/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019. Sebastian Rosenzweig
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019 Sebastian Rosenzweig (sebastian.rosenzweig@med.uni-goettingen.de)

 */

#include "misc/cppwrap.h"
#include "misc/mri.h"

struct laplace_conf {

	int nn; 	 // number of nearest neighbours
	float sigma;	 // Standard deviation

};

extern const struct laplace_conf laplace_conf_default;

extern void calc_laplace(const struct laplace_conf* conf, const long L_dims[2], complex float* L, const long src_dims[2], const complex float* src);
