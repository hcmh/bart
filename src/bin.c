/* Copyright 2018. The Regents of the University of California.
 * Copyright 2018. Sebastian Rosenzweig.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Sebastian Rosenzweig
 *
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"


static const char usage_str[] = "<bin-signal> <src> <dst>";
static const char help_str[] =
		"Binning \n";


/* Lable binning:
 * Bin a dimension according to the lable-file <bin-signal>
 *
 * Quadrature binning:
 * Rosenzweig, S., Scholand, N., Holme, H. C. M., & Uecker, M. (2018).
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular Spectrum Analysis (SSA-FARY).
 * arXiv preprint arXiv:1812.09057.

 * n: Number of bins
 * state: Contains amplitudes for both EOFs
 * idx: Bins for 'idx' motion (0: cardiac, 1: respiration)
 *
 */

// Binning by equal central angle
static void det_bins(const complex float* state, const long bins_dims[DIMS], float* bins, const int idx, const unsigned int n)
{
	unsigned int T = bins_dims[TIME_DIM];
	float central_angle = 2. * M_PI / n;
	for (unsigned int t=0; t<T; t++) {
		bins[idx * T + t] = floor( ( M_PI + atan2f(crealf(state[t]), crealf(state[T + t]))) / central_angle );
 		//debug_printf(DP_INFO, "%f: bin %f\n", (M_PI + atan2f(crealf(state[t]), crealf(state[T + t]))) * 360 / 2. / M_PI, bins[idx * T + t]);
	}
}


// Calculate maximum number of samples in a bin
static int get_binsize_max(const long bins_dims[DIMS], const float* bins, unsigned int n_card, unsigned int n_resp)
{
	// Array to count number of appearances of a bin
	long count_dims[2] = { n_card, n_resp };
	int* count = md_alloc(2, count_dims, sizeof(int));
	md_clear(2, count_dims, count, sizeof(int));

	unsigned int T = bins_dims[TIME_DIM]; // Number of time samples
	for (unsigned int t=0; t<T; t++) { // Iterate through time
		int cBin = (int) bins[0 * T + t];
		int rBin = (int) bins[1 * T + t];
		count[rBin * n_card + cBin]++;
	}

	// Determine value of array maximum
	int binsize_max = 0;
	for (unsigned int r=0; r<n_resp; r++) {
		for(unsigned int c=0; c<n_card; c++) {
			if (count[r * n_card + c] > binsize_max)
				binsize_max = count[r * n_card + c];
			//debug_printf(DP_INFO, "%d\n", count[r * n_card + c]);
		}
	}

	md_free(count);

	return binsize_max;
}

// Copy spokes from input array to correct position in output array
static void asgn_bins(const long bins_dims[DIMS], const float* bins, const long sg_dims[DIMS], complex float* sg, const long in_dims[DIMS], const complex float* in, const int n_card, const int n_resp)
{
	// Array to keep track of numbers of spokes already asigned to each bin
	long count_dims[2] = { n_card, n_resp };
	int* count = md_alloc(2, count_dims, sizeof(int));
	md_clear(2, count_dims, count, sizeof(int));


	// Array to store a single spoke (including read-out, [coils] and slices)
	long in_singleton_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, in_singleton_dims, in_dims);
	complex float* in_singleton = md_alloc(DIMS, in_singleton_dims, CFL_SIZE);

	unsigned int T = bins_dims[TIME_DIM]; // Number of time samples
	long pos0[DIMS] = { 0 };
	long pos1[DIMS] = { 0 };

	for (unsigned int t = 0; t < T; t++) { // Iterate all spokes of input array
		pos0[TIME_DIM] = t;
		md_copy_block(DIMS, pos0, in_singleton_dims, in_singleton, in_dims, in, CFL_SIZE);

		int cBin = (int) bins[0 * T + t];
		int rBin = (int) bins[1 * T + t];
		pos1[PHS2_DIM] = count[rBin * n_card + cBin]; // free spoke index in respective bin
		pos1[TIME_DIM] = cBin;
		pos1[TIME2_DIM] = rBin;

		md_copy_block(DIMS, pos1, sg_dims, sg, in_singleton_dims, in_singleton, CFL_SIZE);

		count[rBin * n_card + cBin]++;
	}

	md_free(in_singleton);
}

static void moving_average(const long state_dims[DIMS], complex float* state, const unsigned int mavg_window)
{
	// Pad with boundary values
	long pad_dims[DIMS];
	md_copy_dims(DIMS, pad_dims, state_dims);
	pad_dims[TIME_DIM] = state_dims[TIME_DIM] + mavg_window -1;
	complex float* pad = md_alloc(DIMS, pad_dims, CFL_SIZE);
	md_resize_center(DIMS, pad_dims, pad, state_dims, state, CFL_SIZE);

	long singleton_dims[DIMS];
	md_select_dims(DIMS, TIME2_FLAG, singleton_dims, state_dims);
	complex float* singleton = md_alloc(DIMS, singleton_dims, CFL_SIZE);
	long pos[DIMS] = { 0 };
	md_copy_block(DIMS, pos, singleton_dims, singleton, state_dims, state, CFL_SIZE); // Get first value of array

	long start = labs((pad_dims[TIME_DIM] / 2) - (state_dims[TIME_DIM] / 2));
	for (int i = 0; i < start; i++) { // Fill beginning of pad array
		pos[TIME_DIM] = i;
		md_copy_block(DIMS, pos, pad_dims, pad, singleton_dims, singleton, CFL_SIZE);
	}

	long end = mavg_window - start;
	pos[TIME_DIM] = state_dims[TIME_DIM] - 1;
	md_copy_block(DIMS, pos, singleton_dims, singleton, state_dims, state, CFL_SIZE); // Get last value of array

	for (int i = 0; i < end; i++) { // Fill end of pad array
		pos[TIME_DIM] = pad_dims[TIME_DIM] - 1 - i;
		md_copy_block(DIMS, pos, pad_dims, pad, singleton_dims, singleton, CFL_SIZE);
	}

	// Calc moving average
	long tmp_dims[DIMS + 1];
	md_copy_dims(DIMS, tmp_dims, pad_dims);
	tmp_dims[DIMS] = 1;

	long tmp_strs[DIMS + 1];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);

	tmp_dims[TIME_DIM] = state_dims[TIME_DIM]; // Moving average reduced temporal dimension

	long tmp2_strs[DIMS + 1];
	md_calc_strides(DIMS + 1, tmp2_strs, tmp_dims, CFL_SIZE);

	tmp_dims[DIMS] = mavg_window;
	tmp_strs[DIMS] = tmp_strs[TIME_DIM];

	complex float* mavg = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	md_moving_avgz2(DIMS + 1, DIMS, tmp_dims, tmp2_strs, mavg, tmp_strs, pad);

	md_zsub(DIMS, state_dims, state, state, mavg);

	md_free(pad);
	md_free(mavg);
	md_free(singleton);

}

int main_bin(int argc, char* argv[])
{
	unsigned int n_resp = 0;
	unsigned int n_card = 0;
	unsigned int mavg_window = 0;
	bool quadrature_binning = false;
	int cluster_dim = -1;

	long resp_lables_idx[2] = { 0, 1 };
	long card_lables_idx[2] = { 2, 3 };




	const struct opt_s opts[] = {
		OPT_INT('d', &cluster_dim, "dim", "Label Binning: Cluster dimension"),
		OPT_UINT('R', &n_resp, "n_resp", "Quadrature Binning: Number of respiratory lables"),
		OPT_UINT('C', &n_card, "n_card", "Quadrature Binning: Number of cardiac lables"),
		OPT_VEC2('r', &resp_lables_idx, "x:y", "(Respiration: Eigenvector index)"),
		OPT_VEC2('c', &card_lables_idx, "x:y", "(Cardiac motion: Eigenvector index)"),
		OPT_UINT('a', &mavg_window, "window", "Quadrature Binning: Moving average"),

	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	// Input
	long lables_dims[DIMS];
	complex float* lables = load_cfl(argv[1], DIMS, lables_dims);

	long src_dims[DIMS];
	complex float* src = load_cfl(argv[2], DIMS, src_dims);

	if (n_resp > 0 || n_card > 0) {

		quadrature_binning = true;
		assert(n_resp > 0 && n_card > 0);

	} else
		if (cluster_dim < 0 || src_dims[cluster_dim] != 1) // Dimension to store data for each cluster must be empty
			error("Choose empty cluster dimension!");


	if (quadrature_binning) {

		if (lables_dims[TIME_DIM] < 2)
			error("Check dimensions of lables array!");

		// Extract respiratory lables
		long resp_state_dims[DIMS];
		md_copy_dims(DIMS, resp_state_dims, lables_dims);
		resp_state_dims[TIME2_DIM] = 2;
		complex float* resp_state = md_alloc(DIMS, resp_state_dims, CFL_SIZE);

		long resp_state_singleton_dims[DIMS];
		md_copy_dims(DIMS, resp_state_singleton_dims, resp_state_dims);
		resp_state_singleton_dims[TIME2_DIM] = 1;
		complex float* resp_state_singleton = md_alloc(DIMS, resp_state_singleton_dims, CFL_SIZE);

		long pos[DIMS] = { 0 };
		for (int i = 0; i < 2; i++){

			pos[TIME2_DIM] = resp_lables_idx[i];
			md_copy_block(DIMS, pos, resp_state_singleton_dims, resp_state_singleton, lables_dims, lables, CFL_SIZE);
			pos[TIME2_DIM] = i;
			md_copy_block(DIMS, pos, resp_state_dims, resp_state, resp_state_singleton_dims, resp_state_singleton, CFL_SIZE);

		}


		// Extract cardiac lables
		long card_state_dims[DIMS];
		md_copy_dims(DIMS, card_state_dims, lables_dims);
		card_state_dims[TIME2_DIM] = 2;
		complex float* card_state = md_alloc(DIMS, card_state_dims, CFL_SIZE);

		long card_state_singleton_dims[DIMS];
		md_copy_dims(DIMS, card_state_singleton_dims, card_state_dims);
		card_state_singleton_dims[TIME2_DIM] = 1;
		complex float* card_state_singleton = md_alloc(DIMS, card_state_singleton_dims, CFL_SIZE);

		for (int i = 0; i < 2; i++){

			pos[TIME2_DIM] = card_lables_idx[i];
			md_copy_block(DIMS, pos, card_state_singleton_dims, card_state_singleton, lables_dims, lables, CFL_SIZE);
			pos[TIME2_DIM] = i;
			md_copy_block(DIMS, pos, card_state_dims, card_state, card_state_singleton_dims, card_state_singleton, CFL_SIZE);

		}

		if (mavg_window > 0) {

			moving_average(resp_state_dims, resp_state, mavg_window);
			moving_average(card_state_dims, card_state, mavg_window);

		}

		// Array to store bin-index for samples
		long bins_dims[DIMS];
		md_copy_dims(DIMS, bins_dims, lables_dims);
		bins_dims[TIME2_DIM] = 2; // Respiration and Cardiac motion
		float* bins = md_alloc(DIMS, bins_dims, FL_SIZE);

		// Determine bins
		det_bins(resp_state, bins_dims, bins, 1, n_resp); // respiratory motion
		det_bins(card_state, bins_dims, bins, 0, n_card); // cardiac motion

		int binsize_max = get_binsize_max(bins_dims, bins, n_card, n_resp);

		// Binned data
		long binned_dims[DIMS];
		md_copy_dims(DIMS, binned_dims, src_dims);
		binned_dims[TIME_DIM] = n_card;
		binned_dims[TIME2_DIM] = n_resp;
		binned_dims[PHS2_DIM] = binsize_max;
		complex float* binned = create_cfl(argv[3], DIMS, binned_dims);
		md_clear(DIMS, binned_dims, binned, CFL_SIZE);

		// Assign to bins
		asgn_bins(bins_dims, bins, binned_dims, binned, src_dims, src, n_card, n_resp);

		md_free(bins);
		md_free(card_state);
		md_free(card_state_singleton);

		md_free(resp_state);
		md_free(resp_state_singleton);

		unmap_cfl(DIMS, binned_dims, binned);

	} else {

		md_check_compat(DIMS, ~0u, src_dims, lables_dims);
		md_check_bounds(DIMS, ~0u, lables_dims, src_dims);

		// Allow only one dimension to be > 1
		long dim; // Dimension to be binned
		long count = 0;
		for (unsigned int i = 0; i < DIMS; i++)
			if (lables_dims[i] > 1) {
				dim = i;
				count++;
			}
		assert(count == 1);

		long N = lables_dims[dim]; // number of samples to be binned

		// Determine number of clusters
		long n_clusters = 0;
		for (int i = 0; i < N; i++)
			if (n_clusters < (long)lables[i])
				n_clusters = (long)lables[i];

		n_clusters += 1; // Account for zero-based indexing

		// Determine all cluster sizes
		long* cluster_size = md_alloc(1, &n_clusters, sizeof(long));
		md_clear(1, &n_clusters, cluster_size, sizeof(long));

		for (int i = 0; i < N; i++)
			cluster_size[(long)lables[i]]++;

		// Determine maximum cluster size
		long cluster_max = 0;

		for (int i = 0; i < n_clusters; i++)
			cluster_max = (cluster_size[i] > cluster_max) ? cluster_size[i] : cluster_max;

		// Initialize output
		long dst_dims[DIMS];
		md_copy_dims(DIMS, dst_dims, src_dims);
		dst_dims[cluster_dim] = cluster_max;
		dst_dims[dim] = n_clusters;
		complex float* dst = create_cfl(argv[3], DIMS, dst_dims);
		md_clear(DIMS, dst_dims, dst, CFL_SIZE);


		// Do binning
		long singleton_dims[DIMS];
		md_select_dims(DIMS, ~MD_BIT(dim), singleton_dims, src_dims);
		complex float* singleton = md_alloc(DIMS, singleton_dims, CFL_SIZE);

		long* idx = md_alloc(1, &n_clusters, sizeof(long));
		md_clear(1, &n_clusters, idx, sizeof(long));
		long pos_src[DIMS] = { 0 };
		long pos_dst[DIMS] = { 0 };

		for (int i = 0; i < N; i++) { // TODO: Speed but by direct copying

			pos_src[dim] = i;
			md_copy_block(DIMS, pos_src, singleton_dims, singleton, src_dims, src, CFL_SIZE);

			pos_dst[dim] = (long)lables[i];
			pos_dst[cluster_dim] = idx[(long)lables[i]]; // Next empty singleton index for i-th cluster

			md_copy_block(DIMS, pos_dst, dst_dims, dst, singleton_dims, singleton, CFL_SIZE);

			idx[(long)lables[i]]++;

			// Debug output
			if (i % (long)(0.1 * N) == 0)
				debug_printf(DP_DEBUG3, "Binning: %f%\n", i*1./N * 100);

		}

		md_free(idx);
		md_free(singleton);

	}

	unmap_cfl(DIMS, lables_dims, lables);
	unmap_cfl(DIMS, src_dims, src);



	exit(0);
}
