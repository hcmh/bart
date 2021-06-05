/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/convcorr.h"

#include "utest.h"

#ifdef USE_CUDNN
const int cudnn_imple = 1;
#else
const int cudnn_imple = 0;
#endif


static bool test_convcorr_frw_cf_1D(void)
{
	enum {N = 6};
	long odims[N] = {2, 1, 3, 1, 1, 4};
	long idims[N] = {1, 5, 5, 1, 1, 4};
	long kdims[N] = {2, 5, 3, 1, 1, 1};

	bool test = test_zconvcorr_fwd(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}
UT_REGISTER_TEST(test_convcorr_frw_cf_1D);

static bool test_convcorr_bwd_in_cf_1D(void)
{
	enum {N = 6};
	long odims[N] = {2, 1, 3, 1, 1, 4};
	long idims[N] = {1, 5, 5, 1, 1, 4};
	long kdims[N] = {2, 5, 3, 1, 1, 1};

	bool test = test_zconvcorr_bwd_in(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}
UT_REGISTER_TEST(test_convcorr_bwd_in_cf_1D);

static bool test_convcorr_bwd_krn_cf_1D(void)
{
	enum {N = 6};
	long odims[N] = {2, 1, 3, 1, 1, 4};
	long idims[N] = {1, 5, 5, 1, 1, 4};
	long kdims[N] = {2, 5, 3, 1, 1, 1};

	bool test = test_zconvcorr_bwd_krn(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}
UT_REGISTER_TEST(test_convcorr_bwd_krn_cf_1D);


static bool test_convcorr_frw_cf_2D(void)
{
	enum {N = 6};
	long odims[N] = {2, 1, 3, 2, 1, 4};
	long idims[N] = {1, 5, 5, 5, 1, 4};
	long kdims[N] = {2, 5, 3, 4, 1, 1};

	bool test = test_zconvcorr_fwd(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}
UT_REGISTER_TEST(test_convcorr_frw_cf_2D);

static bool test_convcorr_bwd_in_cf_2D(void)
{
	enum {N = 6};
	long odims[N] = {2, 1, 3, 2, 1, 4};
	long idims[N] = {1, 5, 5, 5, 1, 4};
	long kdims[N] = {2, 5, 3, 4, 1, 1};

	bool test = test_zconvcorr_bwd_in(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}
UT_REGISTER_TEST(test_convcorr_bwd_in_cf_2D);

static bool test_convcorr_bwd_krn_cf_2D(void)
{
	enum {N = 6};
	long odims[N] = {2, 1, 3, 2, 1, 4};
	long idims[N] = {1, 5, 5, 5, 1, 4};
	long kdims[N] = {2, 5, 3, 4, 1, 1};

	bool test = test_zconvcorr_bwd_krn(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}
UT_REGISTER_TEST(test_convcorr_bwd_krn_cf_2D);

static bool test_convcorr_frw_cf(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 1, 3, 2, 1, 4 };
	long idims[N] = { 1, 5, 5, 5, 4, 4 };
	long kdims[N] = { 2, 5, 3, 4, 4, 1 };

	bool test = test_zconvcorr_fwd(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_frw_cf);


static bool test_convcorr_bwd_in_cf(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 1, 3, 2, 1, 4 };
	long idims[N] = { 1, 5, 5, 5, 4, 4 };
	long kdims[N] = { 2, 5, 3, 4, 4, 1 };

	bool test = test_zconvcorr_bwd_in(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_bwd_in_cf);


static bool test_convcorr_bwd_krn_cf(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 1, 3, 2, 1, 4 };
	long idims[N] = { 1, 5, 5, 5, 4, 4 };
	long kdims[N] = { 2, 5, 3, 4, 4, 1 };

	bool test = test_zconvcorr_bwd_krn(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_bwd_krn_cf);


static bool test_convcorr_frw_rand_ord(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 4, 3, 1, 2, 1 };
	long idims[N] = { 1, 4, 5, 5, 5, 4 };
	long kdims[N] = { 2, 1, 3, 5, 4, 4 };

	bool test = test_zconvcorr_fwd(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					52, NULL, NULL, false,
					1.e-6, 0, 0 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_frw_rand_ord);


static bool test_convcorr_bwd_in_rand_ord(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 4, 3, 1, 2, 1 };
	long idims[N] = { 1, 4, 5, 5, 5, 4 };
	long kdims[N] = { 2, 1, 3, 5, 4, 4 };

	bool test = test_zconvcorr_bwd_in(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					52, NULL, NULL, false,
					1.e-6, 0, 0 + cudnn_imple);

	UT_ASSERT(test);
}


UT_REGISTER_TEST(test_convcorr_bwd_in_rand_ord);

static bool test_convcorr_bwd_krn_rand_ord(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 4, 3, 1, 2, 1 };
	long idims[N] = { 1, 4, 5, 5, 5, 4 };
	long kdims[N] = { 2, 1, 3, 5, 4, 4 };

	bool test = test_zconvcorr_bwd_krn(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					52, NULL, NULL, false,
					1.e-6, 0, 0 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_bwd_krn_rand_ord);


static bool test_convcorr_frw_cf_one_channel(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 1, 3, 2, 1, 4 };
	long idims[N] = { 1, 1, 5, 5, 4, 4 };
	long kdims[N] = { 2, 1, 3, 4, 4, 1 };

	bool test = test_zconvcorr_fwd(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_frw_cf_one_channel);


static bool test_convcorr_bwd_in_cf_one_channel(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 1, 3, 2, 1, 4 };
	long idims[N] = { 1, 1, 5, 5, 4, 4 };
	long kdims[N] = { 2, 1, 3, 4, 4, 1 };

	bool test = test_zconvcorr_bwd_in(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_bwd_in_cf_one_channel);


static bool test_convcorr_bwd_krn_cf_one_channel(void)
{
	enum { N = 6 };
	long odims[N] = { 2, 1, 3, 2, 1, 4 };
	long idims[N] = { 1, 1, 5, 5, 4, 4 };
	long kdims[N] = { 2, 1, 3, 4, 4, 1 };

	bool test = test_zconvcorr_bwd_krn(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, NULL, NULL, false,
					1.e-6, 2, 2 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_bwd_krn_cf_one_channel);


static bool test_convcorr_frw_cf_dil_strs(void)
{
	enum { N = 6 };
	long odims[N] = 	{ 2, 1, 3, 3, 1, 4 };
	long idims[N] = 	{ 1, 5, 5, 7, 4, 4 };
	long kdims[N] = 	{ 2, 5, 2, 3, 4, 1 };

	long dilation[N] =	{ 1, 1, 2, 1, 1, 1 };
	long strides[N] =	{ 1, 1, 1, 2, 1, 1 };

	bool test = test_zconvcorr_fwd(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, dilation, strides, false,
					1.e-6, 0, 1 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_frw_cf_dil_strs);


static bool test_convcorr_bwd_in_cf_dil_strs(void)
{
	enum { N = 6 };
	long odims[N] = 	{ 2, 1, 3, 3, 1, 4 };
	long idims[N] = 	{ 1, 5, 5, 7, 4, 4 };
	long kdims[N] = 	{ 2, 5, 2, 3, 4, 1 };

	long dilation[N] =	{ 1, 1, 2, 1, 1, 1 };
	long strides[N] =	{ 1, 1, 1, 2, 1, 1 };

#ifdef NON_DETERMINISTIC
	int gpu_count = 1;
#else
	int gpu_count = 0;
#endif


	bool test = test_zconvcorr_bwd_in(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, dilation, strides, false,
					1.e-6, 0, gpu_count); //no deterministic cudnn algorithm

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_bwd_in_cf_dil_strs);


static bool test_convcorr_bwd_krn_cf_dil_strs(void)
{
	enum { N = 6 };
	long odims[N] = 	{ 2, 1, 3, 3, 1, 4 };
	long idims[N] = 	{ 1, 5, 5, 7, 4, 4 };
	long kdims[N] = 	{ 2, 5, 2, 3, 4, 1 };

	long dilation[N] =	{ 1, 1, 2, 1, 1, 1 };
	long strides[N] =	{ 1, 1, 1, 2, 1, 1 };

	bool test = test_zconvcorr_bwd_krn(	N,
					odims, MD_STRIDES(N, odims, CFL_SIZE),
					idims, MD_STRIDES(N, idims, CFL_SIZE),
					kdims, MD_STRIDES(N, kdims, CFL_SIZE),
					28, dilation, strides, false,
					1.e-6, 0, 1 + cudnn_imple);

	UT_ASSERT(test);
}

UT_REGISTER_TEST(test_convcorr_bwd_krn_cf_dil_strs);

