
#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "T1fun.h"
#include "noir/model.h"


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
	complex float* Gamma;
	complex float* R1s;

    complex float* tmp_map;
    complex float* tmp_ones;
    complex float* tmp_data;

	complex float* TI;
    
    float scaling_Gamma;
    float scaling_R1s;

};

DEF_TYPEID(T1_s);

// Calculate Model: Mss * (1 - (1 + scaling_Gamma * Gamma)) * exp(-t.*scaling_R1s.*R1s)
static void T1_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
    long pos[data->N];

    for (int i = 0; i < data->N; i++)
            pos[i] = 0;
    
    // Mss                                                                 
    pos[COEFF_DIM] = 0;
    md_copy_block(data->N, pos, data->map_dims, data->Mss, data->in_dims, src, CFL_SIZE); 	

    // Gamma = M0/Mss                                                                 
    pos[COEFF_DIM] = 1;
    md_copy_block(data->N, pos, data->map_dims, data->Gamma, data->in_dims, src, CFL_SIZE); 	

    // R1s                                                                 
    pos[COEFF_DIM] = 2;
    md_copy_block(data->N, pos, data->map_dims, data->R1s, data->in_dims, src, CFL_SIZE); 	

    // exp(-t.*scaling_R1s*R1s)                                             
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, -1);
    md_zexp(data->N, data->out_dims, dst, dst);
    
    // 1 + scaling_Gamma*Gamma
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->Gamma, data->scaling_Gamma);
    md_zfill(data->N, data->map_dims, data->tmp_ones, 1.0);
	md_zadd(data->N, data->map_dims, data->tmp_map, data->tmp_ones, data->tmp_map);

    // 1 -(1 + scaling_Gamma*Gamma).*exp(-t.*scaling_R1s*R1s)
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, dst);
    md_zsub2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_ones, data->out_strs, dst);
	
    // Mss*(1 -(1 + scaling_Gamma*Gamma).*exp(-t.*scaling_R1s*R1s))
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Mss, data->out_strs, dst);

}

static void T1_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
    long pos[data->N];

    for (int i = 0; i < data->N; i++)
            pos[i] = 0;

    // ----------calculating Gamma'-----------
    // exp(-t.*scaling_R1s.*R1s)
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);

    // Gamma' = -Mss.*scaling_Gamma.*exp(-t.*scaling_R1s.*R1s)
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -data->scaling_Gamma);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->map_strs, data->Mss);

    // tmp = dGamma
    pos[COEFF_DIM] = 1;
    md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE); 	
    
    // dst = Gamma' * dGamma
    md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // -----------calculating Mss'----------
    // exp(-t.*scaling_R1s.*R1s)
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);    
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    
    // Mss' = 1 - (1 + scaling_Gamma * Gamma) * exp(-t.*scaling_R1s.*R1s) 
    md_zfill(data->N, data->map_dims, data->tmp_ones, 1.0);
    md_zsmul(data->N, data->map_dims, data->tmp_map, data->Gamma, data->scaling_Gamma);
    md_zadd(data->N, data->map_dims, data->tmp_map, data->tmp_map, data->tmp_ones);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_data);

    // tmp = dMss
    pos[COEFF_DIM] = 0;
    md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE); 	

    // dst = dst + dMss * Mss'
    md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // -----------calculating R1s'----------
    // exp(-t.*scaling_R1s.*R1s)
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);    
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    
    // (1 + scaling_Gamma.*Gamma).*scaling_R1s.*Mss
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->Gamma, data->scaling_Gamma);
	md_zadd(data->N, data->map_dims, data->tmp_map, data->tmp_ones, data->tmp_map);
    md_zmul(data->N, data->map_dims, data->tmp_map, data->Mss, data->tmp_map);
    md_zsmul(data->N, data->map_dims, data->tmp_map, data->tmp_map, data->scaling_R1s);
 
    // R1s' = (1 + scaling_Gamma.*Gamma)* scaling_R1s.*Mss.*exp(-t.*scaling_R1s.*R1s) * t
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
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
    
    // ----------calculating Gamma'-----------
    // exp(-t.*scaling_R1s.*R1s)
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);

    // Gamma' = -Mss.*scaling_Gamma.*exp(-t.*scaling_R1s.*R1s)
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -data->scaling_Gamma);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->map_strs, data->Mss);
   
    // conj(Gamma') * src
    md_zconj(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->out_strs, src);

    // sum (conj(M0') * src, t)
    md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE); 
    md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // dst[1] = sum (conj(M0') * src, t) 
    pos[COEFF_DIM] = 1;
    md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE); 	
    
    // -----------calculating Mss'----------
    // exp(-t.*scaling_R1s.*R1s)
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);    
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    
    // Mss' = 1 - (1 + scaling_Gamma * Gamma) * exp(-t.*scaling_R1s.*R1s) 
    md_zfill(data->N, data->map_dims, data->tmp_ones, 1.0);
    md_zsmul(data->N, data->map_dims, data->tmp_map, data->Gamma, data->scaling_Gamma);
    md_zadd(data->N, data->map_dims, data->tmp_map, data->tmp_map, data->tmp_ones);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_data);
    
    // conj(Mss') * src
    md_zconj(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->out_strs, src);
    
    // sum (conj(Mss') * src, t)
    md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE); 
    md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    
    // dst[0] = sum (conj(Mss') * src, t) 
    pos[COEFF_DIM] = 0;
    md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE); 	
    
    // -----------calculating R1s'----------
    // exp(-t.*scaling_R1s.*R1s)
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, data->scaling_R1s);    
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->TI_strs, data->TI);
    md_zsmul(data->N, data->out_dims, data->tmp_data, data->tmp_data, -1);
    md_zexp(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    
    // (1 + scaling_Gamma.*Gamma).*scaling_R1s.*Mss
    md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->Gamma, data->scaling_Gamma);
	md_zadd(data->N, data->map_dims, data->tmp_map, data->tmp_ones, data->tmp_map);
    md_zmul(data->N, data->map_dims, data->tmp_map, data->Mss, data->tmp_map);
    md_zsmul(data->N, data->map_dims, data->tmp_map, data->tmp_map, data->scaling_R1s);
 
    // R1s' = (1 + scaling_Gamma.*Gamma)* scaling_R1s.*Mss.*exp(-t.*scaling_R1s.*R1s) * t
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->TI_strs, data->TI, data->out_strs, data->tmp_data);
    
    // conj(R1s') * src
    md_zconj(data->N, data->out_dims, data->tmp_data, data->tmp_data);
    md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_data, data->out_strs, data->tmp_data, data->out_strs, src);
    
    // real(sum (conj(R1s') * src, t)))
    md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE); 
    md_zadd2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->map_strs, data->tmp_map, data->out_strs, data->tmp_data);
    md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
    
    // dst[2] = real(sum (conj(R1s') * src, t)) 
    pos[COEFF_DIM] = 2;
    md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE); 	
}

static void T1_del(const nlop_data_t* _data)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

	md_free(data->Mss);
	md_free(data->Gamma);
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


struct nlop_s* nlop_T1_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N], const complex float* TI, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif
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
	data->Mss = my_alloc(N, map_dims, CFL_SIZE);
	data->Gamma = my_alloc(N, map_dims, CFL_SIZE);
	data->R1s = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_map = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_ones = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_data = my_alloc(N, out_dims, CFL_SIZE);
	data->TI = my_alloc(N, TI_dims, CFL_SIZE);
    md_copy(N, TI_dims, data->TI, TI, CFL_SIZE); 
    
    data->scaling_Gamma = 1;
    data->scaling_R1s = 0.5;
	
    return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T1_fun, T1_der, T1_adj, NULL, NULL, T1_del);
}

