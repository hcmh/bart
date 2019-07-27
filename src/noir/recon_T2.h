/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef RECON_T2_H
#define RECON_T2_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

#include "recon.h"



extern void T2_recon(const struct noir_conf_s* conf, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* pattern, const _Complex float* mask, const _Complex float* TI, const _Complex float* kspace_datai, _Bool usegpu);

#endif
