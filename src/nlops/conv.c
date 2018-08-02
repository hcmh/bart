/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "linops/someops.h"
#include "linops/linop.h"

#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/nlop.h"

#include "conv.h"


static const struct nlop_s* nlop_fftc_create(int N, const long dims[N], unsigned int flags, bool inv)
{
	auto lfft = (inv ? linop_ifftc_create : linop_fftc_create)(N, dims, flags);
	auto nfft = nlop_from_linop(lfft);
	linop_free(lfft);
	return nfft;
}

struct nlop_s* nlop_conv_create(int N, unsigned int flags, const long odims[N], const long idims1[N], const long idims2[N])
{
	auto nl = nlop_tenmul_create(N, odims, idims1, idims2);

	auto ffto = nlop_fftc_create(N, odims, flags, true);
	auto nl2 = nlop_chain(nl, ffto);
	nlop_free(ffto);
	nlop_free(nl);

	auto ffti1 = nlop_fftc_create(N, idims1, flags, false);
	auto nl3 = nlop_chain2(ffti1, 0, nl2, 0);
	nlop_free(ffti1);
	nlop_free(nl2);

	auto ffti2 = nlop_fftc_create(N, idims2, flags, false);
	auto nl4 = nlop_chain2(ffti2, 0, nl3, 1);
	nlop_free(ffti2);
	nlop_free(nl3);

	return nl4;
}


