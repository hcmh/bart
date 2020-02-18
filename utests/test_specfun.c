
#include "num/specfun.h"

#include "utest.h"

static bool test_sine_integral(void)
{
	double tests[][2] = {	{ 0.,	0. },
				{ 1.,	0.946083 },
				{ M_PI,	1.851937 },/* Wilbraham-Gibbs constant*/
				{ 10,	1.658348 },
				{ 100,	1.562226 }
	};

	for (unsigned int i = 0; i < ARRAY_SIZE(tests); i++) {

		if ( (Si(tests[i][0]) - tests[i][1]) > 10E-5)
			return 0;

		// Test for Si(-z) = -Si(z)
		if ( (Si(-tests[i][0]) + tests[i][1]) > 10E-5)
			return 0;
	}

	return 1;
}

UT_REGISTER_TEST(test_sine_integral);


