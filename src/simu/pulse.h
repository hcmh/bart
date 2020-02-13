
#ifndef __PULSE_H
#define __PULSE_H

struct simdata_pulse {

	float pulse_length;
	float rf_start;
	float rf_end;
	float flipangle;
	float phase;
	float nl;		/*number of zero crossings to the left of the main loop*/
	float nr; 		/*number of zero crossings to the right of the main loop*/
	float n;		/*max(nl, nr)*/
	float t0;		/*time of main lope: t0 =  = pulse_len / ( 2 + (nl-1)  + (nr-1))*/
	float alpha; 		/*windows of pulse ( 0: normal sinc, 0.5: Hanning, 0.46: Hamming)*/
	float A;		/*offset*/
	float energy_scale;	/*Define energy scale factor*/
	bool pulse_applied;
};

extern const struct simdata_pulse simdata_pulse_defaults;

extern void pulse_create(struct simdata_pulse* pulse, float rf_start, float rf_end, float angle, float phase, float nl, float nr, float alpha);

extern float pulse_energy(const struct simdata_pulse* pulse);
extern float pulse_sinc(const struct simdata_pulse* pulse, float t);

#endif

