/* Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2018-2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */
#ifndef RECON_Pixel_H
#define RECON_Pixel_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

#include "noir/recon.h"

struct modBlochFit;

extern void pixel_recon(const struct noir_conf_s* conf, const struct modBlochFit* fit_para, const long dims[DIMS], complex float* img, const complex float* data, _Bool usegpu);

#endif
