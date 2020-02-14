/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef RECON_Bloch_H
#define RECON_Bloch_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

#include "noir/recon.h"


struct modBlochFit;
struct moba_conf;

extern void bloch_recon(const struct moba_conf* conf, const struct modBlochFit* fit_para, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* pattern, const _Complex float* mask, const _Complex float* kspace_data, _Bool usegpu);

#endif
