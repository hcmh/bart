#include "sms_rf.h"
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

void calc_sms_rf_pulse(int mbFactor, long samples, float* sms, float* ref_sinc, double gamma, double slice_distance, double slice_gradient, bool bssfp) {
    float phase = 0.;

    complex float cref_sinc;
    complex float csms;
    float shift[mbFactor], enc[mbFactor];
    
    for (int k = 0; k<mbFactor;k++)
        shift[k] = (k - (mbFactor - 1.) / 2.) * slice_distance * (-2. * M_PI * gamma * slice_gradient * 1.e-6 );

    for(int j = 0; j < mbFactor; j++) {
        
        for (int k = 0; k<mbFactor;k++)
            enc[k] = bssfp ? 0. : 2. * M_PI * (j*k) / mbFactor;

        float absNorm = 1.;
        
        for (int i = 0; i < samples; i++) {
            cref_sinc = *((complex float*) (ref_sinc + sizeof(complex float)/sizeof(float)*i));

            csms = 0. + 0.*I;
            for (int k = 0; k<mbFactor; k++) {
                phase = carg(cref_sinc) + (-samples/2. + i) * shift[k] + enc[k];
                csms += cabs(cref_sinc) * (cos(phase) + sin(phase)*I);
            }

            *((complex float*) (sms + sizeof(complex float)/sizeof(float)*(j*samples + i))) = csms;
            if (cabs(csms)>absNorm)
                absNorm = cabs(csms);
        }

        for (int i = 0; i < samples; i++) {
            *((complex float*) (sms + sizeof(complex float)/sizeof(float)*(j*samples + i))) /= absNorm;
        }
    }
}

