
#include "num/specfun.h"

#include "utest.h"

static bool test_sine_integral(void)
{	
	// TODO: Add more tests using various values
	return (Si(1.) - 0.946083070367183) < 10E-5;
}

UT_REGISTER_TEST(test_sine_integral);


