#ifndef __SMS_RF_H
#define __SMS_RF_H

#include "misc/dllspec.h"
#include "misc/cppwrap.h"

#ifndef __cplusplus
#ifndef bool
typedef int bool;
enum { false, true };
#endif
#endif

BARTLIB_API void BARTLIB_CALL calc_sms_rf_pulse(int mbFactor, long samples, float* sms, float* ref_sinc, 
double gamma, double slice_distance, double slice_gradient, bool bssfp);

#include "misc/cppwrap.h"

#endif // __SMS_RF_H
