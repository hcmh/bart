
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/nested.h"

#include "seq.h"



/*
 * check ordering and dependencies of sequence events
 */
static bool seq_check(int N, const struct seq_event ev[N])
{
	double time = 0.;

	for (int i = 0; i < N; i++) {

#if 0
		if (ev[i].start < time)
			return false;
#endif
		if (ev[i].mid < ev[i].start)
			return false;

		if (ev[i].end < ev[i].mid)
			return false;

		if (   (NULL != ev[i].dependency) 
		    && (ev[i].start < ev[i].dependency->end))
			return false;

		time = ev[i].start;
	}

	return true;
}





/*
 * evaluate gradient at time t
 */
static void event_sample(double m[3], bool deriv, double t, const struct seq_event* ev)
{
	assert(SEQ_EVENT_GRADIENT == ev->type);

	for (int a = 0; a < 3; a++)
		m[a] = 0.;

	if (ev->start > t)
		return;

	if (ev->end < t)
		return;

	double s = ev->start;
	double e = ev->end;
	double c = ev->mid;

	for (int a = 0; a < 3; a++) {

		if (c > s) {

			double A = ev->grad.ampl[a] / (c - s);

			if (t <= c)
				m[a] = A * (deriv ? 1. : (t - s));
		}

		if (e > c) {

			double B = ev->grad.ampl[a] / (e - c);

			if (c < t)
				m[a] = B * (deriv ? 1. : (e - t));
		}
	}
}


/*
 * evaluate sequence at time t
 */
void seq_sample(double m[3], double t, int N, const struct seq_event ev[N])
{
	for (int a = 0; a < 3; a++)
		m[a] = 0.;

	for (int j = 0; j < N; j++) {

		if (SEQ_EVENT_GRADIENT != ev[j].type)
			continue;

		double m0[3];
		event_sample(m0, false, t, &ev[j]);

		for (int a = 0; a < 3; a++)
			m[a] += m0[a];
	}
}	


/*
 * evaluate slew rate of the  sequence at time t
 */
void seq_slew(double m[3], double t, int N, const struct seq_event ev[N])
{
	for (int a = 0; a < 3; a++)
		m[a] = 0.;

	for (int j = 0; j < N; j++) {

		if (SEQ_EVENT_GRADIENT != ev[j].type)
			continue;

		double m0[3];
		event_sample(m0, true, t, &ev[j]);

		for (int a = 0; a < 3; a++)
			m[a] += m0[a];
	}
}


/*
 * check limits of sequence events
 */
static bool seq_check_limits(int N, const struct seq_event ev[N], const struct seq_system* sys)
{
	for (int i = 0; i < N; i++) {

		NESTED(bool, check, (double x[3], double lim))
		{
			for (int k = 0; k < 3; k++)
				if (fabs(x[k]) > lim)
					return false;

			return true;
		}


		double t[3] = {	ev[i].start, ev[i].mid, ev[i].end };
		double m[3];
		double d[3];

		for (int j = 0; j < 3; j++) {

			seq_sample(m, t[j], N, ev);
			seq_slew(d, t[j], N, ev);

			if (check(m, sys->max_grad_ampl))
				return false;
#if 0
			if (check(d, 1. / sys->inv_slew_rate))
				return false;
#endif
		}
	}
	
	return true;
}



/*
 * compute 0th moment of gradient at time t for a
 * (potentially assymetric) triangle
 */
static void moment(double m0[3], double t, const struct seq_event* ev)
{
	assert(SEQ_EVENT_GRADIENT == ev->type);

	for (int a = 0; a < 3; a++)
		m0[a] = 0.;

	if (ev->start > t)
		return;

	double s = ev->start;
	double e = ev->end;
	double c = ev->mid;

	for (int a = 0; a < 3; a++) {

		if (c > s) {

			double A = ev->grad.ampl[a] / (c - s);

			m0[a] += A * powf((MIN(c, t) - s), 2.) / 2.;
		}

		if (e > c) {

			double B = ev->grad.ampl[a] / (e - c);

			if (t > c) {

				m0[a] += B * powf((e - c), 2.) / 2.;

				if (e > t)
       					m0[a] -= B * powf((e - t), 2.) / 2.;
			}
		}
       }
}


/*
 * sum moments for a list of events
 */
static void moment_sum(double m0[3], double t, int N, const struct seq_event ev[N])
{
	for (int a = 0; a < 3; a++)
		m0[a] = 0.;

	for (int i = 0; i < N; i++) {

		double m[3];
		moment(m, t, &ev[i]);

		for (int a = 0; a < 3; a++)
			m0[a] += m[a];
	}
}


/*
 * Compute gradients on a raster. This also works
 * (i.e. yields correct 0 moment) if the abstract
 * gradients do not start and end on the raster.
 * We integrate over each interval to obtain the
 * average gradient.
 */
void seq_compute_gradients(int M, float gradients[M][3], double dt, int N, const struct seq_event ev[N])
{
	for (int i = 0; i < M; i++) 
		for (int a = 0; a < 3; a++)
			gradients[i][a] = 0.;

	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_GRADIENT != ev->type)
			continue;
	
		double s = ev[i].start;
		double e = ev[i].end;


		/*            |    /
                 *            |   /|
                 *            .../..
                 *            | /  |
                 *            |/   |
                 *            /    |
                 *       ..../|    |
                 *  |____|__/_|____|____|
                 *    0    1    2    3  
                 */

		assert((0. <= s) && (e <= dt * M));

		double om[3];

		for (int a = 0; a < 3; a++)
			om[a] = 0.;

		for (int p = truncf(s / dt); p <= ceilf(e / dt); p++) {

			assert((0 <= p) && (p <= M));

			double m0[3];
			moment(m0, (p + 0.5) * dt, &ev[i]);

			for (int a = 0; a < 3; a++) {

				if (p < M)
					gradients[p][a] += (m0[a] - om[a]);

				om[a] = m0[a];
			}
		}
	}
}


/*
 * setup a trapezoid consists of two superimposed triangles
 *
 *    /\/\
 *   / /\ \
 */
static int trapezoid(struct seq_event ev[2], const struct seq_event* dep, double rise, double flat, const double ampl[3])
{
	double start = 0.;

	assert(0 <= rise);
	assert(0 <= flat);

	if (NULL != dep)
		start = dep->end;

	ev[0].start = start;
	ev[0].mid = start + rise;
	ev[0].end = start + rise + flat;
	ev[0].type = SEQ_EVENT_GRADIENT;
	ev[0].dependency = dep;

	for (int i = 0; i < 3; i++)
		ev[0].grad.ampl[i] = ampl[i];
	
	ev[1].start = start + rise;
	ev[1].mid = start + rise + flat;
	ev[1].end = start + 2. * rise + flat;
	ev[1].type = SEQ_EVENT_GRADIENT;
	ev[1].dependency = NULL;

	for (int i = 0; i < 3; i++)
		ev[1].grad.ampl[i] = ampl[i];

	return 2;
}




struct trapezoid {

	double rise;
	double ampl;
	double flat;
}; 


static double trapezoid_moment(struct trapezoid trap)
{
	return trap.ampl * (trap.flat + 1. * trap.rise);
}



/*
 * compute parameters for symmetric trapezoid with given
 * moment using a given slew rate and maximal amplitude
 *
 * mom = rise_max^2 * slew (triangle)
 */
static struct trapezoid trapezoid_amplitude(double mom, double slew, double ampl_max)
{
	assert(0. < mom);
	double rise_max = sqrtf(2. * mom / slew);	// assuming to limit

	struct trapezoid ret;

	ret.ampl = MAX(ampl_max, rise_max * slew);
	ret.rise = ret.ampl / slew;
	ret.flat = (mom / ret.ampl - ret.rise);

	assert(1.E-10 > fabs(trapezoid_moment(ret) - mom));

	return ret;
}



/*
 * compute parameters for softest symmetric trapezoid
 *
 *
 */
static struct trapezoid trapezoid_softest(double mom, double dur, double ampl_max)
{
	// flat * ampl + (dur - flat) / 2. * ampl = mom

	struct trapezoid ret;

	ret.flat = MAX(0., 2. * fabs(mom) / ampl_max - dur);
	ret.rise = (dur - ret.flat) / 2.;
	ret.ampl = mom / (dur - ret.rise);

	assert(0. < ret.rise);
	assert(1.E-10 > fabs(trapezoid_moment(ret) - mom));

	return ret;
}


struct seq_flash_conf seq_flash_defaults = {

	.TR = 0.010,		// [s]
	.TE = 0.005,		// [s]
	.FA = 10.,		// [deg]
	.BW_adc = 1000.,	// [Hz/pixel]
	.BW_rf = 1000.,		// [Hz]
};


struct seq_system seq_sys_skyra = {

	.name = "Skyra",
	.inv_slew_rate = 5.55-3,	// [s/(T/m)]
	.max_grad_ampl = 24e-3,		// [T/m]
	.min_coil_lead = 100e-6, 	// [s]
	.min_dur_readout_rf = 205e-6,	// [s]
	.sample_time = 1e-6,		// [s]
	.raster_time = 10e-6,		// [s]
};

struct seq_system seq_sys_skyra_whisper = {

	.name = "Skyra (whisper)",
	.inv_slew_rate = 20e-3,
	.max_grad_ampl = 22e-3,
	.min_coil_lead = 100e-6,
	.min_dur_readout_rf = 205e-6,
	.sample_time = 1e-6,
	.raster_time = 10e-6,
};


void seq_flash(int N, struct seq_event ev[N], struct seq_flash_conf conf, const struct seq_system* sys)
{
	assert(12 == N);
	UNUSED(conf);


/*   |    |_|_|    |      |	 |
 * z |.../|.|.|\...|......|......|...
 *   |\_/ | | | \_/|      |	 |
 *   |    | | |    |______|______|
 * x |....|.|.|.../|......|......|\..
 *   |    | | |\_/ |      |	 |
 *   |    | | |    |      |	 |
 * y |....|.|.|....|......|......|...
 *   |    | | |\__/|      |	 |
 *   a    b c d    e      f      g
 *
 * TR = g - a 
 * TE = f - c
 * 1/BW_ADC = g - e
 * 1/BW_RF = d - b
 */

	double adc_time = 1. / conf.BW_adc;
	double rf_time = 1. / conf.BW_rf;

	// 6 trapezoidal gradients

	struct trapezoid slice = {	// slice selction

		.ampl = +1.E-4,
		.rise = 0.0001,
		.flat = rf_time,
	};

	struct trapezoid rdout = {	// readout

		.ampl = +1.E-4,
		.rise = 0.0001,
		.flat = adc_time,
	};
	
	double pre_time1 = conf.TE - (adc_time + rf_time) / 2.;
	double pre_time2 = conf.TR - (adc_time + rf_time + pre_time1);

	// keep it simple for now
	double pre_time = MIN(pre_time1, pre_time2);

//	printf("Timing: rf: %f adc: %f gap %f TE: %f TR: %f\n", rf_time, adc_time, pre_time, conf.TR, conf.TE);  

	assert(0. < pre_time);

	// slice de-/rephasor

	assert(slice.rise < pre_time);

	double sldph_time = pre_time - slice.rise;
	double sldph_mom = -trapezoid_moment(slice) / 2.;
	struct trapezoid sldph = trapezoid_softest(sldph_mom, sldph_time, sys->max_grad_ampl);

	// read dephasor

	assert(rdout.rise < pre_time);

	double rddph_time = pre_time - rdout.rise;
	double rddph_mom = -trapezoid_moment(rdout) / 4.;
	struct trapezoid rddph = trapezoid_softest(rddph_mom, rddph_time, sys->max_grad_ampl);


	// phase encoding

	double pe_mom = rddph_mom; // FIXME
	struct trapezoid peenc = trapezoid_softest(pe_mom, pre_time, sys->max_grad_ampl); 

	int n = 0;

	n += trapezoid(ev +  0, NULL,   sldph.rise, sldph.flat, (double[]){ 0., 0., sldph.ampl });
	n += trapezoid(ev +  2, &ev[1], slice.rise, slice.flat, (double[]){ 0., 0., slice.ampl });
	n += trapezoid(ev +  4, &ev[2], rddph.rise, rddph.flat, (double[]){ rddph.ampl, 0., 0. });
	n += trapezoid(ev +  6, &ev[2], peenc.rise, peenc.flat, (double[]){ 0., peenc.ampl, 0. });
	n += trapezoid(ev +  8, &ev[3], sldph.rise, sldph.flat, (double[]){ 0., 0., sldph.ampl });
	n += trapezoid(ev + 10, &ev[5], rdout.rise, rdout.flat, (double[]){ rdout.ampl, 0., 0. });

//	for (int i = 0; i < n; i++)
//		printf("%d: %f %f %f %f %f %f\n", i, ev[i].start, ev[i].mid, ev[i].end, ev[i].grad.ampl[0], ev[i].grad.ampl[1], ev[i].grad.ampl[2]);

	assert(N == n);
	assert(seq_check(N, ev));
	assert(seq_check_limits(N, ev, sys));

#if 1
	// unit tests

	double mom[3];

	moment(mom, ev[0].mid / 4., &ev[0]);
	assert(1.E-10 > fabs(mom[2] - sldph.ampl * sldph.rise / 32.));

	moment(mom, ev[0].mid / 2., &ev[0]);
	assert(1.E-10 > fabs(mom[2] - sldph.ampl * sldph.rise / 8.));

	moment(mom, ev[0].mid, &ev[0]);
	assert(1.E-10 > fabs(mom[2] - sldph.ampl * sldph.rise / 2.));

	moment(mom, ev[0].end, &ev[0]);
	assert(1.E-10 > fabs(mom[2] - sldph.ampl * (sldph.flat + sldph.rise) / 2.));

	moment(mom, ev[1].end, &ev[1]);
	assert(1.E-10 > fabs(mom[2] - sldph.ampl * (sldph.flat + sldph.rise) / 2.));

	moment_sum(mom, ev[1].end, 2, ev);
	assert(1.E-10 > fabs(mom[2] - sldph.ampl * (sldph.flat + sldph.rise)));
#endif
}


