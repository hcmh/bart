/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2019 Nick Scholand <nick.scholand@med.uni-goettingen.de>
 */


#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "simu/simulation.h"
#include "simu/sim_matrix.h"
#include "simu/epg.h"

#include "nlops/nlop.h"

/*
###################################################################
*******************************************************************
------ Prototype Operator for multiple simulation support ---------
*******************************************************************
###################################################################
*/

struct simFun_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* dims;
	const long* map_dims;
	const long* in_dims;
	const long* out_dims;

	const long* strs;
	const long* map_strs;
	const long* in_strs;
	const long* out_strs;
	const long* input_strs;

	float scale[4];

	//derivatives
	complex float* derivatives;

	complex float* input_b1;
	complex float* input_sliceprofile;
	complex float* input_fa_profile;

	//struct with fitting parameters;

	bool use_gpu;

	int counter;

};

DEF_TYPEID(simFun_s);



static void sim_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	double starttime = timestamp();
	debug_printf(DP_DEBUG2, "Started Forward Calculation\n");

	struct simFun_s* data = CAST_DOWN(simFun_s, _data);

	//-------------------------------------------------------------------
	// Run simulation
	//-------------------------------------------------------------------

	//...
	// - choosing different sequence simulation types -> calling function

	//creates
	complex float* sig;
	complex float* dr1;
	complex float* dr2;
	complex float* dm0;

	//-------------------------------------------------------------------
	// Collect data of Signal
	//-------------------------------------------------------------------

	md_copy(data->N, data->out_dims, dst, sig, CFL_SIZE);
	
	//...

	//-------------------------------------------------------------------
	// Collect data of derivatives in single arrray
	//-------------------------------------------------------------------

	md_clear(data->N, data->dims, data->derivatives, CFL_SIZE);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	md_set_dims(data->N, pos, 0);

	pos[COEFF_DIM] = 0; // R1
	md_copy_block(data->N, pos, data->dims, data->derivatives, data->out_dims, dr1, CFL_SIZE);

	pos[COEFF_DIM] = 2; // R2
	md_copy_block(data->N, pos, data->dims, data->derivatives, data->out_dims, dr2, CFL_SIZE);

	pos[COEFF_DIM] = 1; // M0
	md_copy_block(data->N, pos, data->dims, data->derivatives, data->out_dims, dm0, CFL_SIZE);

	//...

	double totaltime = timestamp() - starttime;
	debug_printf(DP_DEBUG2, "Time = %.2f s\n", totaltime);
}


static void sim_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	// START_TIMER;
	debug_printf(DP_DEBUG3, "Start Derivative\n");

	struct simFun_s* data = CAST_DOWN(simFun_s, _data);

	md_clear(data->N, data->out_dims, dst, CFL_SIZE);

	md_ztenmul(data->N, data->out_dims, dst, data->dims, data->derivatives, data->in_dims, src);

	// PRINT_TIMER("BLOCH: Time of Derivative\n");
}



static void sim_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	// START_TIMER;
	debug_printf(DP_DEBUG3, "Start Adjoint\n");

	struct simFun_s* data = CAST_DOWN(simFun_s, _data);

	md_clear(data->N, data->in_dims, dst, CFL_SIZE);

	md_zfmacc2(data->N, data->dims, data->in_strs, dst, data->out_strs, src, data->strs, data->derivatives);

	// PRINT_TIMER("BLOCH: Time of Adjoint\n");
}


static void sim_del(const nlop_data_t* _data)
{
	struct simFun_s* data = CAST_DOWN(simFun_s, _data);

	md_free(data->derivatives);

	md_free(data->input_b1);
	md_free(data->input_sliceprofile);
	md_free(data->input_fa_profile);

	xfree(data->dims);
	xfree(data->map_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->strs);
	xfree(data->map_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);
	xfree(data->input_strs);

	// xfree(data);
}


static struct nlop_s* nlop_sim_create(int N, const long dims[N], const long map_dims[N], const long out_dims[N], const long in_dims[N], /* struct with sim data,*/ bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif
	PTR_ALLOC(struct simFun_s, data);
	SET_TYPEID(simFun_s, data);
	//-------------------------------------------------------------------
	// Copy all to data struct
	//-------------------------------------------------------------------

	//...

	

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), sim_fun, sim_der, sim_adj, NULL, NULL, sim_del);
}
