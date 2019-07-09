#include <math.h>

#include "misc/debug.h"

#include "simu/bloch.h"
#include "simu/simulation.h"

#include "num/ode.h"

#include "sim_matrix.h"

struct ode_matrix_simu_s {

	unsigned int N;
	void* sim_data;
};


static void ode_matrix_fun_simu(void* _data, float* x, float t, const float* in)
{
	struct ode_matrix_simu_s* data = _data;
	struct SimData* sim_data = data->sim_data;
	
	unsigned int N = data->N;
	
	if( sim_data->pulseData.pulse_applied ){ 
		
		float w1 = sinc_pulse( &sim_data->pulseData, t );
		sim_data->gradData.gb_eff[0] = cosf( sim_data->pulseData.phase ) * w1 + sim_data->gradData.gb[0];
		sim_data->gradData.gb_eff[1] = sinf( sim_data->pulseData.phase ) * w1 + sim_data->gradData.gb[1];
	}
	else{
        sim_data->gradData.gb_eff[0] = sim_data->gradData.gb[0];
		sim_data->gradData.gb_eff[1] = sim_data->gradData.gb[1];
	}
	
	sim_data->gradData.gb_eff[2] = sim_data->gradData.gb[2] + sim_data->voxelData.w;
	
	float matrix_time[N][N];
	bloch_matrix_ode_sa(matrix_time, sim_data->voxelData.r1, sim_data->voxelData.r2, sim_data->gradData.gb_eff);

	for (unsigned int i = 0; i < N; i++) {

		x[i] = 0.;

		for (unsigned int j = 0; j < N; j++)
			x[i] += matrix_time[i][j] * in[j];
	}
}

void ode_matrix_interval_simu(float h, float tol, unsigned int N, float x[N], float st, float end, void* sim_data)
{
	struct ode_matrix_simu_s data = { N, sim_data };
	ode_interval(h, tol, N, x, st, end, &data, ode_matrix_fun_simu);
}


void mat_exp_simu(int N, float t, float out[N][N], void* sim_data)
{
	// compute F(t) := exp(tA)

	// F(0) = id
	// d/dt F = A

	float h = t / 100.;
	float tol = 1.E-6;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++)
			out[i][j] = (i == j) ? 1. : 0.;

		ode_matrix_interval_simu(h, tol, N, out[i], 0., t, sim_data);
	}
}
