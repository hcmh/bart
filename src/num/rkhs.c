
#include <complex.h>

#include "num/linalg.h"

#include "rkhs.h"


void rkhs_matrix(int N, complex float (*km)[N][N], int D, const float (*x)[N][D], krn_t krn, float alpha)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			(*km)[i][j] = krn(D, (*x)[i], (*x)[j]) + ((i == j) ? alpha : 0.);
	
	cholesky(N, *km);
}


void rkhs_matrix2(int N, int M, complex float (*km)[N][N][M][M], int D, const float (*x)[N][D], krn2_t krn, float alpha)
{
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			krn(M, (*km)[i][j], D, (*x)[i], (*x)[j]);

			if (i != j)
				continue;

			for (int s = 0; s < M; s++)
				for (int t = 0; t < M; t++)
					(*km)[i][j][s][t] += alpha;
		}
	}
	
	cholesky(N * M, (void*)&km[0][0][0][0]);
}


void rkhs_cardinal(int N, int D, complex float u[N], const complex float km[N][N], const float x[N][D], const float p[D], krn_t krn)
{
	complex float y[N];

	for (int i = 0; i < N; i++)
		y[i] = krn(D, x[i], p);

	cholesky_solve(N, u, km, y);
}


void rkhs_cardinal2(int N, int M, int D, complex float u[N][M][M], const complex float km[N][N][M][M], const float x[N][D], const float p[D], krn2_t krn)
{
	complex float y[N][M][M];

	for (int i = 0; i < N; i++)
		krn(M, y[i], D, x[i], p);

	cholesky_solve(N * M, (void*)&u[0][0][0], (void*)&km[0][0][0][0], (void*)&y[0][0][0]);
}


