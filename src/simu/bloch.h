
// rad s^-1 T^-1
#define GAMMA_H1 267.513e6  

//  @3T
#define WATER_T1 3.0
#define WATER_T2 0.3

#define SKYRA_B0 3.

// T/m
#define SKYRA_GRADIENT 0.045

// T/m/s
#define SKYRA_RAMP 200.

extern void bloch_ode(float out[3], const float in[3], float m0, float t1, float t2, const float gb[3]);
extern void bloch_relaxation(float out[3], float t, const float in[3], float m0, float t1, float t2, const float gb[3]);
extern void bloch_excitation(float out[3], float t, const float in[3], float m0, float t1, float t2, const float gb[3]);

extern void bloch_matrix_ode(float matrix[4][4], float m0, float t1, float t2, const float gb[3]);

