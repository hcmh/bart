
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

                if (0 == i)                     // first intitial pulse
                        out[i] = 180. / M_PI * in[ind];

                else if (0 != i && 0 == ind)    // fulfill continuous measurement if repetitions > length_pa_array
                        out[i] = 180. / M_PI * (in[ind] + in[i-1]);

                else
                        out[i] = 180. / M_PI * (in[ind] + in[ind-1]);
        }
}

void get_antihsfp_fa(int repetitions, complex float* fa_out) {

        assert(0 < repetitions);

        polar2fa(repetitions, N_PA_ANTIHSFP, fa_out, polarangles_antihsfp);
}