/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"

#include "weights.h"





const struct nlop_s* deflatten_weights_create(const struct nlop_s* network, unsigned long flag)
{
	int count = 0;
	long size_in = 0;
	for (int i = 0; i < nlop_get_nr_in_args(network); i++)
		if(!(MD_IS_SET(flag, i))){

			count += 1;
			size_in += md_calc_size(nlop_generic_domain(network, i)->N, nlop_generic_domain(network, i)->dims);
	}

	assert(0 < count);

	struct nlop_s* result = NULL;
	long pos = 0;

	for(int i = 0, j = 0; i < count; i++){

		while(MD_IS_SET(flag, j))
			j += 1;

		const struct linop_s* lin_tmp = linop_copy_selected_create2(nlop_generic_domain(network, j)->N, nlop_generic_domain(network, j)->dims, nlop_generic_domain(network, j)->strs, size_in, pos);

		pos += md_calc_size(nlop_generic_domain(network, j)->N, nlop_generic_domain(network, j)->dims);

		if(result == NULL)
			result = nlop_from_linop_F(lin_tmp);
		else
			result = nlop_dup_F(nlop_combine_FF(result, nlop_from_linop_F(lin_tmp)), 0, 1);

		j += 1;

	}

	return result;
}

const struct nlop_s* deflatten_weights(const struct nlop_s* network, unsigned long flag)
{
	const struct nlop_s* deflatten = deflatten_weights_create(network, flag);
	const struct nlop_s* result = nlop_combine(network, deflatten);

	int o = nlop_get_nr_out_args(network);
	int count = nlop_get_nr_out_args(deflatten);

	nlop_free(deflatten);

	for(int i = 0, j = 0; i < count; i++){

		while(MD_IS_SET(flag, j))
			j += 1;
			result = nlop_link_F(result, o, j);
	}

	return result;
}

const struct nlop_s* deflatten_weightsF(const struct nlop_s* network, unsigned long flag)
{
	const struct nlop_s* result = deflatten_weights(network, flag);
	nlop_free(network);

	return result;
}
