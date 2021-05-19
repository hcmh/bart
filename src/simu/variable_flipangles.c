
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "misc/mri.h"

#include "polar_angles.h"
#include "variable_flipangles.h"

// Function to convert polar angles [rad] to flipangles [deg] with offset
static void polar2fa(int repetitions, int length_pa_array, complex float* out, float* in) {

        int ind = 0;

        for(int i = 0; i < repetitions; i++) {

                ind = i%length_pa_array;

                out[i] = 180. / M_PI * ((0 == ind) ? in[ind] : (in[ind] + in[ind-1])) + 0 * I;
        }
}

void get_antihsfp_fa(int repetitions, complex float* fa_out) {

        assert(0 < repetitions);

        polar2fa(repetitions, N_PA_ANTIHSFP, fa_out, polarangles_antihsfp);
}