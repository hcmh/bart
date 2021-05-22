
extern void ode_matrix_interval_simu(float h, float tol, unsigned int N, float x[N], float st, float end, void* sim_data);

extern void mat_exp_simu(int N, float t, float out[N][N], void* sim_data);

extern void matrix_bloch_simulation(void* _data, complex float (*mxy_sig)[3], complex float (*sa_r1_sig)[3], complex float (*sa_r2_sig)[3], complex float (*sa_m0_sig)[3], complex float (*sa_b1_sig)[3]);
