
#ifndef __PULSE_H
#define __PULSE_H

#include <stdbool.h>
#include <complex.h>

#include "misc/types.h"

struct pulse {

	TYPEID* TYPEID;

	float flipangle;
	float duration;		/* pulse duration */
	//
	complex float (*eval)(const struct pulse *p, float t);
};

struct pulse_sinc {

	INTERFACE(struct pulse);

	float alpha;		/* windows of pulse (0: normal sinc, 0.5: Hanning, 0.46: Hamming) */
	float A;		/* amplitude */
	float bwtp;		/* BWTP=2*n for n=nl=nr */
	int SMS_multiband; /* multi-band factor for SMS */
	int SMS_partition; 
	float SMS_slice_distance; /* center-to-center slice distance defined by phase modulationn */
	float Gs_amp;	/* amplitude of slice-selective gradient*/

};

extern const struct pulse_sinc pulse_sinc_defaults;

extern void pulse_sinc_init(struct pulse_sinc* ps, float duration, float angle /*[deg]*/, float phase, float bwtp, float alpha, 
int multiband, int partition, float slice_distance, float Gs_amp);
extern float pulse_sinc_integral(const struct pulse_sinc* ps);

inline complex float pulse_eval(const struct pulse* p, float t)
{
	return p->eval(p, t);
}

struct pulse_rect {

	INTERFACE(struct pulse);

	float A;		/* amplitude */
};

extern const struct pulse_rect pulse_rect_defaults;

extern void pulse_rect_init(struct pulse_rect* pr, float duration, float angle /*[deg]*/, float phase);


struct pulse_hypsec {

	INTERFACE(struct pulse);

	float a0;
	float beta;
	float mu;
};

extern const struct pulse_hypsec pulse_hypsec_defaults;

extern float pulse_hypsec_phase(const struct pulse_hypsec* pr, float t);
extern void pulse_hypsec_init(struct pulse_hypsec* pr);

#endif		// __PULSE_H

