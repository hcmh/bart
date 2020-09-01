/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "nlops/nlop.h"
#include "nlops/nltest.h"

#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nn/tf_wrapper.h"
#include "utest.h"

// for convolution: output_dim == input_dim - (kernel_dim - 1)
// for other dimensions: two dimensions are same and other == 1

/**
static bool test_nlop_trans_con(void)
{
	enum { N = 1 };
	long dims_image[N] = {10};
	long dims_kernel[N] = {3};
	long dims_output[N] = {8};
	unsigned int conv_flags = 1;

	complex float* dst_trans = md_alloc(N, dims_output, CFL_SIZE);
	complex float* dst_conv = md_alloc(N, dims_output, CFL_SIZE)
}

*/

static bool test_nn_tf_forward(void)
{
 	auto nlop = nlop_tf_create(1, 1, "/media/radon_home/deep_recon/exported/pixel_cnn");

	nlop_debug(DP_INFO, nlop);

	auto dom = nlop_generic_domain(nlop, 0);
	complex float* in = md_alloc(dom->N, dom->dims, dom->size);
	md_clear(dom->N, dom->dims, in, dom->size);


	auto cod = nlop_generic_codomain(nlop, 0);
	complex float* out = md_alloc(dom->N, dom->dims, dom->size);
	md_zfill(cod->N, cod->dims, out, 5. + 5*I);

	nlop_apply(nlop, cod->N, cod->dims, out, dom->N, dom->dims, in);
	//nlop_generic_apply_unchecked(nlop, 2, {(void*)out, (void*)in});

	printf("Loss : %f + %f i\n", creal(*out), cimag(*out));

	return true;
}

UT_REGISTER_TEST(test_nn_tf_forward);
