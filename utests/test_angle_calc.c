#include <complex.h>
#include <math.h>
#include <stdint.h>

#include<time.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "noncart/anglecalc.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"




static uint64_t xoroshiro128plus(uint64_t s[2])
{
	uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	uint64_t result = s0 + s1;
	s1 ^= s0;
	s[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
	s[1] = (s1 << 37) | (s1 >> 27);
	return result;
}

// https://www.pcg-random.org/posts/bounded-rands.html
static uint32_t bounded_rand(uint32_t range, uint64_t s[2]) {
	// calculates divisor = 2**32 / range
	uint32_t divisor = ((-range) / range) + 1;
	if (divisor == 0) // overflow, it's really 2**32
		return 0;

	while (true) {

		uint32_t r = xoroshiro128plus(s) >> 32;
		uint32_t val = r / divisor;
		if (val < range)
			return val;
	}
}



#define ANGLE_CALC_N 1e6
// #define ANGLE_CALC_N 1e9
static bool test_angle_calc(void)
{


	uint64_t random_state[2];
	random_state[0] = 0x18778b5b42b6fec7;
	random_state[1] = 0x2c8afdaa7cb21bec;

#if 0
	srand(time(0));
	random_state[0] = rand();
	random_state[1] = rand();
#endif

	double TOL = UT_TOL*1e-5;

	for (int i = 0; i < ANGLE_CALC_N; ++i) {

		long excitation = bounded_rand(512, random_state);
		long echo = bounded_rand(16, random_state);
		long repetition = bounded_rand(4096, random_state);
		long inversion_repetition = bounded_rand(192, random_state);
		long slice = bounded_rand(192, random_state);
		enum ePEMode mode = 1+bounded_rand(12, random_state);
		long num_slices = slice + bounded_rand(192, random_state);
		long num_turns = 1+bounded_rand(50, random_state);
		long start_pos_GA = bounded_rand(128, random_state);
		long mb_factor  = 1+bounded_rand(32, random_state);
		long num_echoes = 1 + echo + bounded_rand(32, random_state);
		long num_inv_repets = 1 + inversion_repetition + bounded_rand(128, random_state);
		long lines_to_measure = 1 + excitation + bounded_rand(2048, random_state);
		long repetitions_to_measure = 1 + repetition + bounded_rand(4096, random_state);
		bool double_angle = bounded_rand(2, random_state);


		double ph_ref = dgetRotAngle_ref(excitation, echo, repetition, inversion_repetition, slice, mode,
			num_slices, num_turns, start_pos_GA, mb_factor, num_echoes, num_inv_repets, lines_to_measure,
			repetitions_to_measure, double_angle);

		double ph = dgetRotAngle(excitation, echo, repetition, inversion_repetition, slice, mode,
			num_slices, num_turns, start_pos_GA, mb_factor, num_echoes, num_inv_repets, lines_to_measure,
			repetitions_to_measure, double_angle);

		if (!safe_isfinite(fabs(ph_ref - ph))) {

			debug_printf(DP_INFO, "%ld\t%ld\t%ld\t%ld\t%ld\t%d\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%d\n", excitation, echo, repetition, inversion_repetition, slice, mode,
						num_slices, num_turns, start_pos_GA, mb_factor, num_echoes, num_inv_repets, lines_to_measure,
					       repetitions_to_measure, double_angle);
			debug_printf(DP_INFO, "ref:%.6e\t ang: %.6e\t diff: %.9f\t tol: %.9f\n",ph_ref, ph, fabs(ph_ref - ph), TOL);
			break;
		}

// 		debug_printf(DP_INFO, "%d\n", i);
		if (fabs(ph_ref - ph) > TOL) {
// 			debug_printf(DP_INFO, "%d\t%ld\t%d\t%d\n", mode, num_turns, mb_factor, double_angle);
// 			debug_printf(DP_INFO, "ref:%.6e\t ang: %.6e\t diff: %.9f\t tol: %.9f\n",ph_ref, ph, fabs(ph_ref - ph), TOL);
			return false;
		}
	}

	return true;
// 	return (err < UT_TOL);
}


UT_REGISTER_TEST(test_angle_calc);

