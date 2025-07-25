
#ifndef _VECOPS_H
#define _VECOPS_H

extern const struct vec_ops cpu_ops;

struct vec_ops {

	void (*float2double)(long N, double* dst, const float* src);
	void (*double2float)(long N, float* dst, const double* src);
	double (*dot)(long N, const float* vec1, const float* vec2);
	double (*asum)(long N, const float* vec);
	void (*zsum)(long N, _Complex float* vec);
	double (*zl1norm)(long N, const _Complex float* vec);

	_Complex double (*zdot)(long N, const _Complex float* vec1, const _Complex float* vec2);

	void (*axpy)(long N, float* a, float alpha, const float* x);
	void (*axpbz)(long N, float* out, const float a, const float* x, const float b, const float* z);

	void (*pow)(long N, float* dst, const float* src1, const float* src2);
	void (*sqrt)(long N, float* dst, const float* src);
	void (*round)(long N, float* dst, const float* src);

	void (*zle)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*le)(long N, float* dst, const float* src1, const float* src2);

	void (*add)(long N, float* dst, const float* src1, const float* src2);
	void (*sub)(long N, float* dst, const float* src1, const float* src2);
	void (*mul)(long N, float* dst, const float* src1, const float* src2);
	void (*div)(long N, float* dst, const float* src1, const float* src2);
	void (*fmac)(long N, float* dst, const float* src1, const float* src2);
	void (*fmacD)(long N, double* dst, const float* src1, const float* src2);
	void (*smul)(long N, float alpha, float* dst, const float* src1);
	void (*sadd)(long N, float alpha, float* dst, const float* src1);

	void (*zmul)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zdiv)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmac)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmacD)(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zmulc)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmacc)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfmaccD)(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zfsq2)(long N, _Complex float* dst, const _Complex float* src1);

	void (*zsmul)(long N, _Complex float val, _Complex float* dst, const _Complex float* src1);
	void (*zsadd)(long N, _Complex float val, _Complex float* dst, const _Complex float* src1);

	void (*zpow)(long N,  _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zphsr)(long N, _Complex float* dst, const _Complex float* src);
	void (*zconj)(long N, _Complex float* dst, const _Complex float* src);
	void (*zexpj)(long N, _Complex float* dst, const _Complex float* src);
	void (*zexp)(long N, _Complex float* dst, const _Complex float* src);
	void (*zlog)(long N, _Complex float* dst, const _Complex float* src);
	void (*zarg)(long N, _Complex float* dst, const _Complex float* src);
	void (*zabs)(long N, _Complex float* dst, const _Complex float* src);
	void (*zatanr)(long N, _Complex float* dst, const _Complex float* src);
	void (*zatan2r)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);

	void (*exp)(long N, float* dst, const float* src);
	void (*log)(long N, float* dst, const float* src);

	void (*zsin)(long N, _Complex float* dst, const _Complex float* src);
	void (*zcos)(long N, _Complex float* dst, const _Complex float* src);
	void (*zacosr)(long N, _Complex float* dst, const _Complex float* src);

	void (*zsinh)(long N, _Complex float* dst, const _Complex float* src);
	void (*zcosh)(long N, _Complex float* dst, const _Complex float* src);

	void (*zcmp)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zdiv_reg)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda);
	void (*zfftmod)(long N, _Complex float* dst, const _Complex float* src, int n, _Bool inv, double phase);

	void (*zmax)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zsmax)(long N, float alpha, _Complex float* dst, const _Complex float* src);
	void (*zsmin)(long N, float alpha, _Complex float* dst, const _Complex float* src);

	void (*smax)(long N, float val, float* dst, const float* src1);
	void (*max)(long N, float* dst, const float* src1, const float* src2);
	void (*min)(long N, float* dst, const float* src1, const float* src2);

	void (*zsoftthresh_half)(long N, float lambda,  _Complex float* dst, const _Complex float* src);
	void (*zsoftthresh)(long N, float lambda,  _Complex float* dst, const _Complex float* src);
	void (*softthresh_half)(long N, float lambda,  float* dst, const float* src);
	void (*softthresh)(long N, float lambda,  float* dst, const float* src);
//	void (*swap)(long N, float* a, float* b);
	void (*zhardthresh)(long N, int k, _Complex float* d, const _Complex float* x);
	void (*zhardthresh_mask)(long N, int k, _Complex float* d, const _Complex float* x);

	void (*pdf_gauss)(long N, float mu, float sig, float* dst, const float* src);

	void (*real)(long N, float* dst, const _Complex float* src);
	void (*imag)(long N, float* dst, const _Complex float* src);
	void (*zcmpl_real)(long N, _Complex float* dst, const float* src);
	void (*zcmpl_imag)(long N, _Complex float* dst, const float* src);
	void (*zcmpl)(long N, _Complex float* dst, const float* real_src, const float* imag_src);

	void (*zfill)(long N, _Complex float val, _Complex float* dst);

	void (*zsetnanzero)(long N, _Complex float* dst, const _Complex float* src);
};

#endif

