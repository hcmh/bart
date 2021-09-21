/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/specfun.h"

#include "misc/nested.h"
#include "misc/misc.h"

#include "noncart/grid.h"

#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

//#define USE_FXDIV
#ifdef USE_FXDIV
#include "num/fxdiv.h"
#endif

#include "noncart/gpu_grid.h"

// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024

static int blocksize(int N)
{
	return BLOCKSIZE;
}

static long gridsize(long N)
{
	return (N + BLOCKSIZE - 1) / BLOCKSIZE;
}


static void cuda_error(int line, cudaError_t code)
{
	const char *err_str = cudaGetErrorString(code);
	error("cuda error: %d %s \n", line, err_str);
}

#define CUDA_ERROR(x)	({ cudaError_t errval = (x); if (cudaSuccess != errval) cuda_error(__LINE__, errval); })


// Linear interpolation
__device__ static __inline__ float lerp(float a, float b, float c)
{
	return (1. - c) * a + c * b;
}

// Linear interpolation look up
__device__ static float intlookup(int n, const float* table, float x)
{
	int index = (int)(x * (n - 1));
	float fpart = x * (n - 1) - (float)index;
	float l = lerp(table[index], table[index + 1], fpart);
	return l;
}

enum { kb_size = 100 };
static float* kb_table = NULL;
static double kb_beta = -1.;

static void kb_precompute_gpu(double beta)
{
	// precompute kaiser bessel table
	#pragma	omp critical
	if (-1 == kb_beta) {

		float table_cpu[kb_size + 1];
		kb_precompute(beta, kb_size, table_cpu);
		kb_beta = beta;

		CUDA_ERROR(cudaMalloc(&kb_table, (1 + kb_size) * sizeof(float)));
		CUDA_ERROR(cudaMemcpy(kb_table, table_cpu, (1 + kb_size) * sizeof(float), cudaMemcpyDefault));
	}

	assert(fabs(kb_beta - beta) < 1.E-6);
}

#define MAX_GRID_DIMS 3

struct grid_data {

	int N;
	float os;
	float width;
	bool periodic;

	long samples;
	long grid_dims[4];
	long ksp_dims[4];

	int ch;
	long off_ch_ksp;
	long off_ch_grid;

	float* kb_table;
};

struct grid_data_device {

	float pos[MAX_GRID_DIMS];
	int pos_grid[MAX_GRID_DIMS];
	int sti[MAX_GRID_DIMS];
	int eni[MAX_GRID_DIMS];
	int off[MAX_GRID_DIMS];
};

__device__ static __inline__ void dev_atomic_zadd_scl(cuFloatComplex* arg, cuFloatComplex val, float scl)
{
	atomicAdd(&(arg->x), val.x * scl);
	atomicAdd(&(arg->y), val.y * scl);
}

__device__ static __inline__ void dev_zadd_scl(cuFloatComplex* arg, cuFloatComplex val, float scl)
{
	arg->x += val.x * scl;
	arg->y += val.y * scl;
}

#if 1
__device__ static void grid_point_r(const struct grid_data* gd, const struct grid_data_device* gdd, int N, long ind, float d, cuFloatComplex* dst, const cuFloatComplex* src)
{
	if (0 == N) {

		//printf("%e %e %e\n", d, src[0].x, src[0].y);

		for (int c = 0; c < gd->ch; c++)
			dev_atomic_zadd_scl(dst + (ind + c * gd->off_ch_grid) , src[c * gd->off_ch_ksp], d);

	} else {

		N--;

		for (int w = gdd->sti[N]; w <= gdd->eni[N]; w++) {

			float frac = fabs(((float)w - gdd->pos[N]));
			float d2 = d * intlookup(kb_size, gd->kb_table, frac / gd->width);
			long ind2 = (ind * gd->grid_dims[N] + ((w + gdd->off[N]) % gd->grid_dims[N]));

			grid_point_r(gd, gdd, N, ind2, d2, dst, src);
		}
	}
}

#else


__device__ static void grid_point_r(const struct grid_data* gd, const struct grid_data_device* gdd, int N, long ind, float d, cuFloatComplex* dst, const cuFloatComplex* src)
{
	float frac;
	assert(3 == N);
	assert(0 == ind);

	for (int z = gdd->sti[2]; z <= gdd->eni[2]; z++) {

		frac = fabs(((float)z - gdd->pos[2]));

		float dz = d * intlookup(kb_size, gd->kb_table, frac / gd->width);
		long off_z = (1 * gd->grid_dims[2] + ((z + gdd->off[2]) % gd->grid_dims[2]));

		for (int y = gdd->sti[1]; z <= gdd->eni[1]; y++) {

			frac = fabs(((float)y - gdd->pos[1]));

			float dy = dz * intlookup(kb_size, gd->kb_table, frac / gd->width);
			long off_y = (off_z * gd->grid_dims[1] + ((y + gdd->off[1]) % gd->grid_dims[1]));

			for (int x = gdd->sti[0]; x <= gdd->eni[0]; x++) {

				frac = fabs(((float)x - gdd->pos[0]));

				float dx = dy * intlookup(kb_size, gd->kb_table, frac / gd->width);
				long off_x = (off_y * gd->grid_dims[0] + ((x + gdd->off[0]) % gd->grid_dims[0]));

				for (int c = 0; c < gd->ch; c++)
					dev_atomic_zadd_scl(dst + (off_x + c * gd->off_ch_grid) , src[c * gd->off_ch_ksp], dx);

			}
		}
	}
}
#endif
// loop over out-dims and krn-dims and copy elements from input (copies one element per thread)
__global__ static void kern_grid(struct grid_data conf, const cuFloatComplex* traj, cuFloatComplex* grid, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	struct grid_data_device gdd;

	for (long i = start; i < conf.samples; i += stride) {

		for (int j = 0; j < conf.N; j++) {

			gdd.pos[j] = conf.os * (traj[i * 3 + j]).x;
			gdd.pos[j] += (conf.grid_dims[j] > 1) ? ((float) conf.grid_dims[j] / 2.) : 0.;

			gdd.sti[j] = (int)ceil(gdd.pos[j] - conf.width);
			gdd.eni[j] = (int)floor(gdd.pos[j] + conf.width);
			gdd.off[j] = 0;

			if (gdd.sti[j] > gdd.eni[j])
				continue;

			if (!conf.periodic) {

				gdd.sti[j] = MAX(gdd.sti[j], 0);
				gdd.eni[j] = MIN(gdd.eni[j], conf.grid_dims[j] - 1);

			} else {

				while (gdd.sti[j] + gdd.off[j] < 0)
					gdd.off[j] += conf.grid_dims[j];
			}

			if (1 == conf.grid_dims[j]) {

				assert(0. == gdd.pos[j]); // ==0. fails nondeterministically for test_nufft_forward bbdec08cb
				gdd.sti[j] = 0;
				gdd.eni[j] = 0;
			}
		}

		//printf("sample: %ld; %d %d %d; %d %d %d; %f %f %f\n", i, gdd.sti[0], gdd.sti[1], gdd.sti[2], gdd.eni[0], gdd.eni[1], gdd.eni[2], gdd.pos[0], gdd.pos[1], gdd.pos[2]);

		grid_point_r(&conf, &gdd, conf.N, 0, 1., grid, src + i);
	}
}


void cuda_grid(const struct grid_conf_s* conf, const _Complex float* traj, const long grid_dims[4], _Complex float* grid, const long ksp_dims[4], const _Complex float* src)
{

	kb_precompute_gpu(conf->beta);

	struct grid_data gd = {

		.N = 3,//(1 == grid_dims[2]) ? 2 : 3,
		.os = conf->os,
		.width = conf->width,
		.periodic = conf->periodic,

		.samples = ksp_dims[1] * ksp_dims[2],

		.grid_dims = { grid_dims[0], grid_dims[1], grid_dims[2], grid_dims[3]},
		.ksp_dims = { ksp_dims[0], ksp_dims[1], ksp_dims[2], ksp_dims[3]},

		.ch = (int)ksp_dims[3],

		.off_ch_ksp = md_calc_size(3, ksp_dims),
		.off_ch_grid = md_calc_size(3, grid_dims),

		.kb_table = kb_table,
	};

	kern_grid<<<gridsize(gd.samples), blocksize(gd.samples)>>>(gd, (const cuFloatComplex*)traj, (cuFloatComplex*)grid, (const cuFloatComplex*)src);
}

// loop over out-dims and krn-dims and copy elements from input (copies one element per thread)
__global__ static void kern_grid2(struct grid_data conf, const cuFloatComplex* traj, cuFloatComplex* grid, const cuFloatComplex* src)
{
	long xstart = threadIdx.x + blockDim.x * blockIdx.x;
	long xstride = blockDim.x * gridDim.x;

	long ystart = threadIdx.y + blockDim.y * blockIdx.y;
	long ystride = blockDim.y * gridDim.y;

	long zstart = threadIdx.z + blockDim.z * blockIdx.z;
	long zstride = blockDim.z * gridDim.z;

	struct grid_data_device gdd;
	for (gdd.pos_grid[2] = zstart; gdd.pos_grid[2] < conf.grid_dims[2]; gdd.pos_grid[2] += zstride)
	for (gdd.pos_grid[1] = ystart; gdd.pos_grid[1] < conf.grid_dims[1]; gdd.pos_grid[1] += ystride)
	for (gdd.pos_grid[0] = xstart; gdd.pos_grid[0] < conf.grid_dims[0]; gdd.pos_grid[0] += xstride) {

		for (long i = 0; i < conf.samples; i++) {

			bool grid_point = true;
			float weight = 1.;

			for (int j = 0; grid_point && (j < conf.N); j++) {

				gdd.pos[j] = conf.os * (traj[i * 3 + j]).x;
				gdd.pos[j] += (conf.grid_dims[j] > 1) ? ((float) conf.grid_dims[j] / 2.) : 0.;

				gdd.sti[j] = (int)ceil(gdd.pos[j] - conf.width);
				gdd.eni[j] = (int)floor(gdd.pos[j] + conf.width);
				gdd.off[j] = 0;

				if (gdd.sti[j] > gdd.eni[j])
					continue;

				if (!conf.periodic) {

					gdd.sti[j] = MAX(gdd.sti[j], 0);
					gdd.eni[j] = MIN(gdd.eni[j], conf.grid_dims[j] - 1);

				} else {

					while (gdd.sti[j] > gdd.pos_grid[j] + gdd.off[j])
						gdd.off[j] += conf.grid_dims[j];

					while (gdd.eni[j] < gdd.pos_grid[j] + gdd.off[j])
						gdd.off[j] -= conf.grid_dims[j];
				}

				if (1 == conf.grid_dims[j]) {

					assert(0. == gdd.pos[j]); // ==0. fails nondeterministically for test_nufft_forward bbdec08cb
					gdd.sti[j] = 0;
					gdd.eni[j] = 0;
				}

				grid_point = (gdd.sti[j] <= gdd.pos_grid[j] + gdd.off[j]) && (gdd.pos_grid[j] <= gdd.eni[j]);

				if (grid_point) {

					float frac = fabs(((float)(gdd.pos_grid[j] + gdd.off[j]) - gdd.pos[j]));
					weight *= intlookup(kb_size, conf.kb_table, frac / conf.width);
				}
			}

			if (grid_point) {

				for (int c = 0; c < conf.ch; c++)
					dev_zadd_scl(grid + gdd.pos_grid[0] + conf.grid_dims[0] * (gdd.pos_grid[1] + conf.grid_dims[1] * gdd.pos_grid[2]) + c * conf.off_ch_grid, src[i + c * conf.off_ch_ksp], weight);
			}
		}
	}
}


void cuda_grid2(const struct grid_conf_s* conf, const _Complex float* traj, const long grid_dims[4], _Complex float* grid, const long ksp_dims[4], const _Complex float* src)
{
	kb_precompute_gpu(conf->beta);

	struct grid_data gd = {

		.N = 3,
		.os = conf->os,
		.width = conf->width,
		.periodic = conf->periodic,

		.samples = ksp_dims[1] * ksp_dims[2],

		.grid_dims = { grid_dims[0], grid_dims[1], grid_dims[2], grid_dims[3]},
		.ksp_dims = { ksp_dims[0], ksp_dims[1], ksp_dims[2], ksp_dims[3]},

		.ch = (int)ksp_dims[3],

		.off_ch_ksp = md_calc_size(3, ksp_dims),
		.off_ch_grid = md_calc_size(3, grid_dims),

		.kb_table = kb_table,
	};

	dim3 blockDim(16, 16);
    	dim3 gridDim(gd.grid_dims[0] / 16, gd.grid_dims[1] / 16);

	kern_grid2<<<gridDim, blockDim>>>(gd, (const cuFloatComplex*)traj, (cuFloatComplex*)grid, (const cuFloatComplex*)src);
}