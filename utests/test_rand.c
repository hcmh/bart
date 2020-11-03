/* Copyright 2015-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdint.h>

#include "num/rand.h"
#include "misc/debug.h"


#include "utest.h"


static bool test_rand(void)
{
	uint64_t rand_seed[1] = { 476342442 };

	uint32_t a_rand[4] = { 3563222799, 1262491113, 4000861964, 4157953354 };

	for (int i = 0; i < 4; i++) {
		
		uint32_t tmp = rand_spcg32(rand_seed);
		if (tmp != a_rand[i]) 
			return false;
	}

	return true;
}


UT_REGISTER_TEST(test_rand);

