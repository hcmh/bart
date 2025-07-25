
#ifndef _SEQ_EVENT_H
#define _SEQ_EVENT_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

#include "seq/gradient.h"

#define MAX_EVENTS 2048

enum seq_event_type { SEQ_EVENT_PULSE, SEQ_EVENT_GRADIENT, SEQ_EVENT_ADC, SEQ_EVENT_WAIT };

enum rf_type_t { UNDEFINED, EXCITATION, REFOCUSSING, STORE };


struct seq_pulse {

	int shape_id;
	enum rf_type_t type;
	double fa;

	double freq;
	double phase;
};

struct seq_gradient {

	double ampl[3];
};

struct seq_adc {

	long dwell_ns;
	long columns;
	long pos[DIMS];

	double os;

	double freq;
	double phase;
};

struct seq_wait {

};


struct seq_event {

	double start;
	double mid;
	double end;

	enum seq_event_type type;

	const struct seq_event* dependency;

	union {

		struct seq_pulse pulse;
		struct seq_gradient grad;
		struct seq_adc adc;
		struct seq_wait wait;
	};
};



int events_counter(enum seq_event_type type, int N, const struct seq_event ev[__VLA(N)]);
int events_idx(int n, enum seq_event_type type, int N, const struct seq_event ev[__VLA(N)]);

int seq_grad_to_event(struct seq_event ev[2], double start, const struct grad_trapezoid* grad, double proj[3]);

void events_get_te(int E, long te[__VLA(E)], int N, const struct seq_event ev[__VLA(N)]);

double events_end_time(int N, const struct seq_event ev[__VLA(N)], int gradients_only, int flat_end);

void moment(double m0[3], double t, const struct seq_event* ev);
void moment_sum(double m0[3], double t, int N, const struct seq_event ev[__VLA(N)]);

#include "misc/cppwrap.h"

#endif // _SEQ_EVENT_H
