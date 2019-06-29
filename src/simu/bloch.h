
// rad s^-1 T^-1
#define GAMMA_H1 267.513e6

//  @3T
#define WATER_T1 3.0
#define WATER_T2 0.3

#define CSF_T1 3.12
#define CSF_T2 0.16

#define SKYRA_B0 3.

// T/m
#define SKYRA_GRADIENT 0.045

// T/m/s
#define SKYRA_RAMP 200.

#define PI 3.141592653589793

extern void bloch_ode(float out[3], const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_relaxation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_excitation(float out[3], float t, const float in[3], float r1, float r2, const float gb[3]);

extern void bloch_matrix_ode(float matrix[4][4], float r1, float r2, const float gb[3]);
extern void bloch_matrix_int(float matrix[4][4], float t, float r1, float r2, const float gb[3]);

extern void bloch_pdy(float out[3][3], const float in[3], float r1, float r2, const float gb[3]);
extern void bloch_pdp(float out[2][3], const float in[3], float r1, float r2, const float gb[3]);

struct PulseData;

float get_pulse_energy(void* pulseData);

float sinc_pulse(void * pulseData, float t);

float si(float x);
long factorial (int k);
