
extern void ode_matrix_interval_simu(float h, float tol, unsigned int N, float x[N], float st, float end, void* sim_data);

extern void mat_exp_simu(int N, float t, float out[N][N], void* sim_data);

extern void matrix_bloch_simulation(void* _data, float (*mxyOriSig)[3], float (*saT1OriSig)[3], float (*saT2OriSig)[3], float (*densOriSig)[3], complex float* input_sp);
