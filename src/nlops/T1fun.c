
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

    complex float* tmp_map;
    complex float* tmp_ones;
    complex float* tmp_data;

	complex float* TI;
    
    float scaling_M0;
    float scaling_R1s;

};

DEF_TYPEID(T1_s);

// Calculate Model: Mss - (Mss + M0) * exp(-t.*R1s)
static void T1_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
    long pos[data->N];

    for (int i = 0; i < data->N; i++)
            pos[i] = 0;
    
    // Mss                                                                 
    pos[COEFF_DIM] = 0;
    md_copy_block(data->N, pos, data->map_dims, data->Mss, data->in_dims, src, CFL_SIZE); 	

    // M0                                                                 
    pos[COEFF_DIM] = 1;
    md_copy_block(data->N, pos, data->map_dims, data->M0, data->in_dims, src, CFL_SIZE); 	

    // R1s                                                                 
    pos[COEFF_DIM] = 2;
    md_copy_block(data->N, pos, data->map_dims, data->R1s, data->in_dims, src, CFL_SIZE); 	

    // scaling_R1s.*R1s
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);

    // exp(-t.*scaling_R1s*R1s)                                             
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, -1);
    md_zexp(data->N, data->out_dims, dst, dst);
    
    // scaling_M0.*M0
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->M0, data->scaling_M0);
    
    // Mss + scaling_M0*M0
	md_zadd(data->N, data->map_dims, data->tmp_map, data->Mss, data->tmp_map);

    // -(Mss + scaling_M0*M0).*exp(-t.*scaling_R1s*R1s)
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, dst);
    md_zsmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, -1);
	
    // Mss -(Mss + scaling_M0*M0).*exp(-t.*scaling_R1s*R1s)
    md_zadd2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Mss, data->out_strs, dst);

}

static void T1_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
    long pos[data->N];

    for (int i = 0; i < data->N; i++)
            pos[i] = 0;

    // scaling_R1s.*R1s
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);

    // M0' = -scaling_M0.*exp(-t.*scaling_R1s.*R1s)
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -data->scaling_M0);

    // tmp = dM0
    pos[COEFF_DIM] = 1;
    md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE); 	
    
    // dst = M0' * dM0
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // Mss' = 1 - exp(-t.*scaling_R1s.*R1s)                                             
    md_zfill(data->N, data->map_dims, data->tmp_ones, 1.0);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, 1.0/data->scaling_M0);
    md_zadd2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_data);

    // tmp = dMss
    pos[COEFF_DIM] = 0;
    md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE); 	

    // dst = dst + dMss * Mss'
    md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // scaling_M0:*exp(-t.*scaling_R1s.*R1s)
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zadd2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_data);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, data->scaling_M0);
    
    // scaling_M0.*M0
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->M0, data->scaling_M0);
    
    // Mss + scaling_M0*M0
	md_zadd(data->N, data->map_dims, data->tmp_ones, data->Mss, data->tmp_map);
   
    // R1s' = (Mss + scaling_M0*M0) * scaling_M0.*exp(-t.*scaling_R1s.*R1s) * t
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->TI_strs, data->TI, data->out_strs, data->tmp_data);
    
    // tmp =  dR1s                                             
    pos[COEFF_DIM] = 2;
    md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE); 	
   
    // dst = dst + dR1s * R1s'
    md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
}

static void T1_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

    long pos[data->N];

    for (int i = 0; i < data->N; i++)
            pos[i] = 0;
    
    // scaling_R1s.*R1s
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_ones, data->map_strs, data->R1s, data->scaling_R1s);
    
    // M0' = -scaling_M0.*exp(-t.*scaling_R1s.*R1s)
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -data->scaling_M0);
    
    // conj(M0') * src
    md_zconj(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->out_strs, src);

    // sum (conj(M0') * src, t)
    md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE); 
    md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // dst[1] = sum (conj(M0') * src, t) 
    pos[COEFF_DIM] = 1;
    md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE); 	
    
    // Mss' = 1 - exp(-t.*scaling_R1s.*R1s)                                             
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zfill(data->N, data->map_dims, data->tmp_ones, 1.0);
    md_zadd2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_data);
    
    // conj(Mss') * src
    md_zconj(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->out_strs, src);
    
    // sum (conj(Mss') * src, t)
    md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE); 
    md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // dst[0] = sum (conj(Mss') * src, t) 
    pos[COEFF_DIM] = 0;
    md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE); 	
    
    // scaling_M0.*M0
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->M0, data->scaling_M0);
    
    // Mss + scaling_M0*M0
	md_zadd(data->N, data->map_dims, data->tmp_ones, data->Mss, data->tmp_map);
   
    // scaling_R1s.*R1s
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);

    // exp(-t.*scaling_R1s.*R1s)                                             
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);

    // R1s' = (Mss + scaling_M0*M0) * scaling_M0.*exp(-t.*scaling_R1s.*R1s) * t
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->TI_strs, data->TI, data->out_strs, data->tmp_data);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, data->scaling_M0);
    
    // conj(R1s') * src
    md_zconj(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->out_strs, src);
    
    // sum (conj(R1s') * src, t)
    md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE); 
    md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    //md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
    
    // dst[2] = sum (conj(R1s') * src, t) 
    pos[COEFF_DIM] = 2;
    md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE); 	
}

static void T1_del(const nlop_data_t* _data)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

	md_free(data->Mss);
	md_free(data->M0);
	md_free(data->R1s);

	md_free(data->TI);

	md_free(data->tmp_map);
	md_free(data->tmp_ones);
	md_free(data->tmp_data);

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


struct nlop_s* nlop_T1_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N], const complex float* TI)
{
	PTR_ALLOC(struct T1_s, data);
	SET_TYPEID(T1_s, data);


	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);
	
	PTR_ALLOC(long[N], nidims);
    md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);
    
	PTR_ALLOC(long[N], ntidims);
    md_copy_dims(N, *ntidims, TI_dims);
	data->TI_dims = *PTR_PASS(ntidims);
	
	PTR_ALLOC(long[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
    data->map_strs = *PTR_PASS(nmstr);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
    data->out_strs = *PTR_PASS(nostr);
    
	PTR_ALLOC(long[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
    data->in_strs = *PTR_PASS(nistr);
	
	PTR_ALLOC(long[N], ntistr);
    md_calc_strides(N, *ntistr, TI_dims, CFL_SIZE);
    data->TI_strs = *PTR_PASS(ntistr);
    
    data->N = N;
	data->Mss = md_alloc(N, map_dims, CFL_SIZE);
	data->M0 = md_alloc(N, map_dims, CFL_SIZE);
	data->R1s = md_alloc(N, map_dims, CFL_SIZE);
	data->tmp_map = md_alloc(N, map_dims, CFL_SIZE);
	data->tmp_ones = md_alloc(N, map_dims, CFL_SIZE);
	data->tmp_data = md_alloc(N, out_dims, CFL_SIZE);
	data->TI = md_alloc(N, TI_dims, CFL_SIZE);
    md_copy(N, TI_dims, data->TI, TI, CFL_SIZE); 
    
//    data->TI = load_cfl("/home/xwang/IR_scripts/TI0", N, TI_dims);
    data->scaling_M0 = 2.0;
    data->scaling_R1s = 1.0;
	
    return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T1_fun, T1_der, T1_adj, NULL, NULL, T1_del);
}

