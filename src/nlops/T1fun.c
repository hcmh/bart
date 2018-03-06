
#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "T1fun.h"


struct T1_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* map_dims;
	const long* TI_dims;
	const long* in_dims;
	const long* out_dims;
	
    const long* map_strs;
	const long* TI_strs;
	const long* in_strs;
	const long* out_strs;

    // Parameter maps
	complex float* Mss;
	complex float* M0;
	complex float* R1s;
    complex float* tmp;

    // Time vector
	complex float* TI;
};

DEF_TYPEID(T1_s);

// Calculate: Mss - (M1 + Mss) * exp(-t.*R1s)
static void T1_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
    long pos[data->N];

    for (int i = 0; i < data->N; i++)
            pos[i] = data->map_dims[i];
    
    // Mss                                                                 
    pos[MAPS_DIM] = 0;
    md_copy_block(data->N, pos, data->map_dims, data->Mss, data->in_dims, src, CFL_SIZE); 	

    // M0                                                                 
    pos[MAPS_DIM] = 1;
    md_copy_block(data->N, pos, data->map_dims, data->M0, data->in_dims, src, CFL_SIZE); 	

    // R1s                                                                 
    pos[MAPS_DIM] = 2;
    md_copy_block(data->N, pos, data->map_dims, data->R1s, data->in_dims, src, CFL_SIZE); 	

    // exp(-t.*R1s)                                             
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->R1s, data->TI_strs, data->TI);
    md_zsmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, -1);
    md_zexp(data->N, data->out_dims, dst, dst);
    
    // Mss + M0
	md_zadd(data->N, data->map_dims, data->tmp, data->Mss, data->M0);

    // -(Mss + M0).*exp(-t.*R1s)
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp, data->out_strs, dst);
    md_zsmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, -1);
	
    // Mss - (Mss + M0).*exp(-t.*R1s)
    md_zadd2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Mss, data->out_strs, dst);
}

static void T1_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
//	md_zmul(data->N, data->dims, dst, src, data->xn);
}

static void T1_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
//	md_zmulc(data->N, data->dims, dst, src, data->xn);
}

static void T1_del(const nlop_data_t* _data)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

	md_free(data->Mss);
	md_free(data->M0);
	md_free(data->R1s);
	md_free(data->TI);
	md_free(data->tmp);

	xfree(data->map_dims);
	xfree(data->TI_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->map_strs);
	xfree(data->TI_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	xfree(data);
}


struct nlop_s* nlop_T1_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], 
                const long TI_dims[N], const complex float* TI)
{
	PTR_ALLOC(struct T1_s, data);
	SET_TYPEID(T1_s, data);

	PTR_ALLOC(long[N], ndims);

	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);
	
	md_copy_dims(N, *ndims, out_dims);
	data->out_dims = *PTR_PASS(ndims);
	
    md_copy_dims(N, *ndims, in_dims);
	data->in_dims = *PTR_PASS(ndims);
    
    md_copy_dims(N, *ndims, TI_dims);
	data->TI_dims = *PTR_PASS(ndims);
	
	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, map_dims, CFL_SIZE);
    data->map_strs = *PTR_PASS(nostr);

	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
    data->out_strs = *PTR_PASS(nostr);
    
	md_calc_strides(N, *nostr, in_dims, CFL_SIZE);
    data->in_strs = *PTR_PASS(nostr);
	
    md_calc_strides(N, *nostr, TI_dims, CFL_SIZE);
    data->TI_strs = *PTR_PASS(nostr);
    
    data->N = N;
	data->Mss = md_alloc(N, map_dims, CFL_SIZE);
	data->M0 = md_alloc(N, map_dims, CFL_SIZE);
	data->R1s = md_alloc(N, map_dims, CFL_SIZE);
	data->tmp = md_alloc(N, map_dims, CFL_SIZE);
	data->TI = md_alloc(N, TI_dims, CFL_SIZE);
    md_copy(N, TI_dims, data->TI, TI, CFL_SIZE); 
	
    return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T1_fun, T1_der, T1_adj, NULL, NULL, T1_del);
}

