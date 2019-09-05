#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "T1MOLLI.h"

//#define general
//#define mphase

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
	complex float* M0;
	complex float* R1;
	complex float* R1s;

	complex float* tmp_map;
	complex float* tmp_map1;
	complex float* tmp_ones;
	complex float* tmp_exp;

	complex float* tmp_dM0;
	complex float* tmp_dR1;
	complex float* tmp_dR1s;

	complex float* TI;

//	float scaling_R1;
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

	// M0
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->M0, data->in_dims, src, CFL_SIZE);

	// R1
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->R1, data->in_dims, src, CFL_SIZE);

	// R1s
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->R1s, data->in_dims, src, CFL_SIZE);

	// -1*scaling_R1s.*R1s
	md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->R1s, -1.0*data->scaling_R1s);

	// exp(-t.*scaling_R1s*R1s):

	for(int k=0; k < (data->TI_dims[5]); k++)
		md_zsmul2(data->N, data->map_dims, data->out_strs, (void*)data->tmp_exp + data->out_strs[5] * k, data->map_strs, (void*)data->tmp_map, data->TI[k]);

	md_zexp(data->N, data->out_dims, data->tmp_exp, data->tmp_exp);

	//1 + R1/R1*
	md_zdiv_reg(data->N, data->map_dims, data->tmp_map, data->R1, data->R1s, 1e-4);
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->tmp_map, 1./data->scaling_R1s);

	md_zfill(data->N, data->map_dims, data->tmp_ones, 1.0);

        md_zadd(data->N, data->map_dims, data->tmp_map1, data->tmp_ones, data->tmp_map);

	// (1 + R1/R1*).*exp(-t.*scaling_R1s*R1s)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1s, data->map_strs, data->tmp_map1, data->out_strs, data->tmp_exp);

	//Model: M0*( R1/R1* -(1 + R1/R1*).*exp(-t.*scaling_R1s*R1s))
	md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_dM0, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dR1s);
	
        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->M0, data->out_strs, data->tmp_dM0);

	// Calculating derivatives
        // R1'
	md_zdiv_reg(data->N, data->map_dims, data->tmp_map1, data->M0, data->R1s, 1e-4);
	md_zsmul(data->N, data->map_dims, data->tmp_map1, data->tmp_map1, 1./data->scaling_R1s);
	md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_dR1, data->map_strs, data->tmp_ones, data->out_strs, data->tmp_exp);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1, data->map_strs, data->tmp_map1, data->out_strs, data->tmp_dR1);

        // R1s'
	long img_dims[data->N];
	md_select_dims(data->N, FFT_FLAGS, img_dims, data->map_dims);

	for (int s=0; s < data->out_dims[13]; s++)
		for(int k=0; k < data->TI_dims[5]; k++)
			//debug_printf(DP_DEBUG2, "\tTI: %f\n", creal(data->TI[k]));
			md_zsmul(data->N, img_dims, (void*)data->tmp_dR1s + data->out_strs[5] * k + data->out_strs[13] * s,
						(void*)data->tmp_dR1s + data->out_strs[5] * k + data->out_strs[13] * s, data->TI[k]);
	
        md_zmul(data->N, data->map_dims, data->tmp_map, data->R1s, data->R1s);
	md_zdiv_reg(data->N, data->map_dims, data->tmp_map, data->R1, data->tmp_map, 1e-4);
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->tmp_map, 1./(data->scaling_R1s*data->scaling_R1s));
	md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->out_strs, data->tmp_exp, data->map_strs, data->tmp_ones);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->map_strs, data->tmp_map, data->out_strs, data->tmp_exp);
        md_zadd(data->N, data->out_dims, data->tmp_dR1s, data->tmp_dR1s, data->tmp_exp);
  
        md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1s, data->map_strs, data->M0, data->out_strs, data->tmp_dR1s);

}

static void T1_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// tmp = dR1
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);

	//const complex float* tmp_M0 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = R1' * dR1
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dR1);

	// tmp = dMss
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	//const complex float* tmp_Mss = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	// dst = dst + dMss * Mss'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dM0);

	// tmp =  dR1s
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	//const complex float* tmp_R1s = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = dst + dR1s * R1s'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dR1s);
}

static void T1_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// sum (conj(M0') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dR1);

	// dst[1] = sum (conj(M0') * src, t)
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// sum (conj(Mss') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dM0);

	// dst[0] = sum (conj(Mss') * src, t)
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// sum (conj(R1s') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dR1s);
	//md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

	// dst[2] = sum (conj(R1s') * src, t)
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}

static void T1_del(const nlop_data_t* _data)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

	md_free(data->R1);
	md_free(data->M0);
	md_free(data->R1s);

	md_free(data->TI);

	md_free(data->tmp_map);
	md_free(data->tmp_ones);
	md_free(data->tmp_exp);

	md_free(data->tmp_dM0);
	md_free(data->tmp_dR1);
	md_free(data->tmp_dR1s);

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


struct nlop_s* nlop_T1MOLLI_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N], const complex float* TI, bool use_gpu)
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
	data->R1 = my_alloc(N, map_dims, CFL_SIZE);
	data->M0 = my_alloc(N, map_dims, CFL_SIZE);
	data->R1s = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_map = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_ones = my_alloc(N, map_dims, CFL_SIZE);
	data->tmp_exp = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_dM0 = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_dR1 = my_alloc(N, out_dims, CFL_SIZE);
	data->tmp_dR1s = my_alloc(N, out_dims, CFL_SIZE);
#ifdef general
	data->TI = my_alloc(N, TI_dims, CFL_SIZE);
#else
	data->TI = md_alloc(N, TI_dims, CFL_SIZE);
#endif
	md_copy(N, TI_dims, data->TI, TI, CFL_SIZE);

//	data->scaling_M0 = 2.0;
	data->scaling_R1s = 1.0;

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T1_fun, T1_der, T1_adj, NULL, NULL, T1_del);
}
