/* Copyright 2018-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
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


/* Reorder binning: [-o]
 * Input a 1D file with <labels> at the <src> dimension, that you want to reorder (according to the label order)
 *
 * ---
 *
 * Lable binning: [-l long]
 * Bin a dimension according to the label-file <bin-signal>
 * The label file must be 1D and the dimension of the <label> file and the <src> file must match at the dimension that you want to bin
 * -d must specify an empty dimension, in which the binned data is stored
 *
 * ---
 *
 * Quadrature binning:
 * Rosenzweig, S., Scholand, N., Holme, H. C. M., & Uecker, M. (2018).
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular Spectrum Analysis (SSA-FARY).
 * arXiv preprint arXiv:1812.09057.
 *
 * n: Number of bins
 * state: Contains amplitudes for both EOFs
 * idx: Bins for 'idx' motion (0: cardiac, 1: respiration)
 *
 */

// Binning by equal central angle
static void det_bins(const complex float* state, const long bins_dims[DIMS], float* bins, const int idx, const int n, const float bin_offset)
{
	int T = bins_dims[TIME_DIM];

	float central_angle = 2. * M_PI / n;
	float offset_angle = (bin_offset == 0.) ? 0 : - bin_offset + central_angle / 2.; // Remove offset

	float angle;

	for (int t = 0; t < T; t++) {

		angle = atan2f(crealf(state[T + t]), crealf(state[t])) + offset_angle;
		angle = (angle < 0) ? (angle + 2. * M_PI) : angle;

		bins[idx * T + t] = floor(angle / central_angle);

 		//debug_printf(DP_INFO, "%f: bin %f\n", (M_PI + atan2f(crealf(state[t]), crealf(state[T + t]))) * 360 / 2. / M_PI, bins[idx * T + t]);
	}
}

/* Check if time is consistent with increasing bin index
 *
 * Idea: Calculate the angles defined by EOF_a & EOF_b (phase diagram!) for time
 * steps total_time/2 and total_time/2+1. If the angle increases, time evolution
 * is consistent with increasing bin-index.  Otherwise, swap EOF_a with EOF_b.
 */

static bool check_valid_time(const long singleton_dims[DIMS], complex float* singleton, const long labels_dims[DIMS], const complex float* labels, const long labels_idx[2])
{
	// Indices at half of total time
	int idx_0 = floor(singleton_dims[TIME_DIM] / 2.);
	int idx_1 = idx_0 + 1;

	long pos[DIMS] = { 0 };

	pos[TIME2_DIM] = labels_idx[0];
	md_copy_block(DIMS, pos, singleton_dims, singleton, labels_dims, labels, CFL_SIZE);

	float a_0 = crealf(singleton[idx_0]);
	float a_1 = crealf(singleton[idx_1]);

	pos[TIME2_DIM] = labels_idx[1];
	md_copy_block(DIMS, pos, singleton_dims, singleton, labels_dims, labels, CFL_SIZE);

	float b_0 = crealf(singleton[idx_0]);
	float b_1 = crealf(singleton[idx_1]);

	float angle_0 = atan2f(b_0, a_0);

	if (angle_0 < 0.)
		angle_0 += 2. * M_PI;

	float angle_1 = atan2f(b_1, a_1);

	if (angle_1 < 0.)
		angle_1 += 2. * M_PI;

	// Check if angle increases (and consider phase wrap!)
	float diff = angle_1 - angle_0;

	return ((diff >= 0.) == (fabsf(diff) <= M_PI));
}


// Calculate maximum number of samples in a bin
static int get_binsize_max(const long bins_dims[DIMS], const float* bins, const int n_card, const int n_resp)
{
	// Array to count number of appearances of a bin
	long count_dims[2] = { n_card, n_resp };

	int* count = md_calloc(2, count_dims, sizeof(int));

	int T = bins_dims[TIME_DIM]; // Number of time samples

	for (int t = 0; t < T; t++) { // Iterate through time

		int cBin = (int)bins[0 * T + t];
		int rBin = (int)bins[1 * T + t];

		count[rBin * n_card + cBin]++;
	}

	// Determine value of array maximum
	int binsize_max = 0;

	for (int r = 0; r < n_resp; r++) {

		for (int c = 0; c < n_card; c++) {

			if (count[r * n_card + c] > binsize_max)
				binsize_max = count[r * n_card + c];

			//debug_printf(DP_INFO, "%d\n", count[r * n_card + c]);
		}
	}

	md_free(count);

	return binsize_max;
}


// Copy spokes from input array to correct position in output array
static void asgn_bins(const long bins_dims[DIMS], const float* bins, const long sg_dims[DIMS], complex float* sg, const long in_dims[DIMS], const complex float* in, const int
n_card, const int n_resp)
{
	// Array to keep track of numbers of spokes already asigned to each bin
	long count_dims[2] = { n_card, n_resp };

	int* count = md_calloc(2, count_dims, sizeof(int));


	// Array to store a single spoke (including read-out, [coils] and slices)
	long in_singleton_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, in_singleton_dims, in_dims);

	complex float* in_singleton = md_alloc(DIMS, in_singleton_dims, CFL_SIZE);

	int T = bins_dims[TIME_DIM]; // Number of time samples

	long pos0[DIMS] = { 0 };
	long pos1[DIMS] = { 0 };

	for (int t = 0; t < T; t++) { // Iterate all spokes of input array

		pos0[TIME_DIM] = t;
		md_copy_block(DIMS, pos0, in_singleton_dims, in_singleton, in_dims, in, CFL_SIZE);

		int cBin = (int)bins[0 * T + t];
		int rBin = (int)bins[1 * T + t];

		pos1[PHS2_DIM] = count[rBin * n_card + cBin]; // free spoke index in respective bin
		pos1[TIME_DIM] = cBin;
		pos1[TIME2_DIM] = rBin;

		md_copy_block(DIMS, pos1, sg_dims, sg, in_singleton_dims, in_singleton, CFL_SIZE);

		count[rBin * n_card + cBin]++;
	}

	md_free(in_singleton);
	md_free(count);
}


static void moving_average(const long state_dims[DIMS], complex float* state, const int mavg_window)
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



static const char usage_str[] = "<label> <src> <dst>";
static const char help_str[] = "Binning\n";



int main_bin(int argc, char* argv[])
{
	unsigned int n_resp = 0;
	unsigned int n_card = 0;
	unsigned int mavg_window = 0;
	unsigned int mavg_window_card = 0;
	float bin_offset = 0.;
	int cluster_dim = -1;

	long resp_labels_idx[2] = { 0, 1 };
	long card_labels_idx[2] = { 2, 3 };

	bool reorder = false;


	const struct opt_s opts[] = {

		OPT_INT('l', &cluster_dim, "dim", "Bin according to labels: Specify cluster dimension"),
		OPT_SET('o', &reorder, "Reorder according to labels"),
		OPT_UINT('R', &n_resp, "n_resp", "Quadrature Binning: Number of respiratory labels"),
		OPT_UINT('C', &n_card, "n_card", "Quadrature Binning: Number of cardiac labels"),
		OPT_VEC2('r', &resp_labels_idx, "x:y", "(Respiration: Eigenvector index)"),
		OPT_VEC2('c', &card_labels_idx, "x:y", "(Cardiac motion: Eigenvector index)"),
		OPT_UINT('a', &mavg_window, "window", "Quadrature Binning: Moving average"),
		OPT_UINT('A', &mavg_window_card, "window", "(Quadrature Binning: Cardiac moving average window)"),
		OPT_FLOAT('O', &bin_offset, "angle", "Quadrature Binning Offset"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	// Input
	long labels_dims[DIMS];
	complex float* labels = load_cfl(argv[1], DIMS, labels_dims);

	long src_dims[DIMS];
	complex float* src = load_cfl(argv[2], DIMS, src_dims);

	char c;

	// Identify binning type
	if ((n_resp > 0) || (n_card > 0)) {

		c = 'Q';

		assert((n_resp > 0) && (n_card > 0));
		assert(cluster_dim == -1);
		assert(!reorder);

	} else if (cluster_dim != -1) {

		c = 'L';

		if ((cluster_dim < 0) || (src_dims[cluster_dim] != 1)) // Dimension to store data for each cluster must be empty
			error("Choose empty cluster dimension!");

		assert(!reorder);
		assert((n_resp == 0) && (n_card == 0));


	} else if (reorder) {

		c = 'R';

		assert((n_resp == 0) && (n_card == 0));
		assert(cluster_dim == -1);

	} else {

		error("Specify binning type!");
	}


	switch (c) {

	case 'Q': { // Quadrature binning

			debug_printf(DP_INFO, "Quadrature binning...\n");

			if (labels_dims[TIME_DIM] < 2)
				error("Check dimensions of labels array!");

			// Extract respiratory labels
			long resp_state_dims[DIMS];
			md_copy_dims(DIMS, resp_state_dims, labels_dims);
			resp_state_dims[TIME2_DIM] = 2;

			complex float* resp_state = md_alloc(DIMS, resp_state_dims, CFL_SIZE);

			long resp_state_singleton_dims[DIMS];
			md_copy_dims(DIMS, resp_state_singleton_dims, resp_state_dims);
			resp_state_singleton_dims[TIME2_DIM] = 1;

			complex float* resp_state_singleton = md_alloc(DIMS, resp_state_singleton_dims, CFL_SIZE);

			bool valid_time_resp = check_valid_time(resp_state_singleton_dims, resp_state_singleton, labels_dims, labels, resp_labels_idx);

			long pos[DIMS] = { 0 };

			for (int i = 0; i < 2; i++){

				pos[TIME2_DIM] = resp_labels_idx[i];
				md_copy_block(DIMS, pos, resp_state_singleton_dims, resp_state_singleton, labels_dims, labels, CFL_SIZE);

				if (valid_time_resp)
					pos[TIME2_DIM] = i;
				else
					pos[TIME2_DIM] = 1 - i;

				md_copy_block(DIMS, pos, resp_state_dims, resp_state, resp_state_singleton_dims, resp_state_singleton, CFL_SIZE);
			}


			// Extract cardiac labels
			long card_state_dims[DIMS];
			md_copy_dims(DIMS, card_state_dims, labels_dims);
			card_state_dims[TIME2_DIM] = 2;

			complex float* card_state = md_alloc(DIMS, card_state_dims, CFL_SIZE);

			long card_state_singleton_dims[DIMS];
			md_copy_dims(DIMS, card_state_singleton_dims, card_state_dims);
			card_state_singleton_dims[TIME2_DIM] = 1;

			complex float* card_state_singleton = md_alloc(DIMS, card_state_singleton_dims, CFL_SIZE);

			bool valid_time_card = check_valid_time(card_state_singleton_dims, card_state_singleton, labels_dims, labels, card_labels_idx);

			for (int i = 0; i < 2; i++) {

				pos[TIME2_DIM] = card_labels_idx[i];
				md_copy_block(DIMS, pos, card_state_singleton_dims, card_state_singleton, labels_dims, labels, CFL_SIZE);

				if (valid_time_card)
					pos[TIME2_DIM] = i;
				else // If time evolution is not consistent with increasing bin-index, swap order of the two EOFs
					pos[TIME2_DIM] = 1 - i;

				md_copy_block(DIMS, pos, card_state_dims, card_state, card_state_singleton_dims, card_state_singleton, CFL_SIZE);
			}

			if (mavg_window > 0) {

				moving_average(resp_state_dims, resp_state, mavg_window);
				moving_average(card_state_dims, card_state, (mavg_window_card > 0) ? mavg_window_card : mavg_window);
			}

#ifdef SSAFARY_PAPER
			dump_cfl("card", DIMS, card_state_dims, card_state);
#endif

			// Array to store bin-index for samples
			long bins_dims[DIMS];
			md_copy_dims(DIMS, bins_dims, labels_dims);
			bins_dims[TIME2_DIM] = 2; // Respiration and Cardiac motion

			float* bins = md_alloc(DIMS, bins_dims, FL_SIZE);

			// Determine bins
			det_bins(resp_state, bins_dims, bins, 1, n_resp, bin_offset); // respiratory motion
			det_bins(card_state, bins_dims, bins, 0, n_card, bin_offset); // cardiac motion

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
		}

		break;

	case 'L': { // Label binning: Bin elements from src according to labels

			debug_printf(DP_INFO, "Label binning...\n");

			md_check_compat(DIMS, ~0u, src_dims, labels_dims);
			md_check_bounds(DIMS, ~0u, labels_dims, src_dims);

			// Allow only one dimension to be > 1
			long dim = 0; // Dimension to be binned
			long count = 0;

			for (unsigned int i = 0; i < DIMS; i++) {

				if (labels_dims[i] > 1) {

					dim = i;
					count++;
				}
			}

			assert(count == 1);

			long N = labels_dims[dim]; // number of samples to be binned

			// Determine number of clusters
			long n_clusters = 0;

			for (int i = 0; i < N; i++)
				if (n_clusters < (long)labels[i])
					n_clusters = (long)labels[i];

			n_clusters += 1; // Account for zero-based indexing

			// Determine all cluster sizes
			long* cluster_size = md_calloc(1, &n_clusters, sizeof(long));

			for (int i = 0; i < N; i++)
				cluster_size[(long)labels[i]]++;

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

			long* idx = md_calloc(1, &n_clusters, sizeof(long));

			long pos_src[DIMS] = { 0 };
			long pos_dst[DIMS] = { 0 };

			for (int i = 0; i < N; i++) { // TODO: Speed but by direct copying

				pos_src[dim] = i;
				md_copy_block(DIMS, pos_src, singleton_dims, singleton, src_dims, src, CFL_SIZE);

				pos_dst[dim] = (long)labels[i];
				pos_dst[cluster_dim] = idx[(long)labels[i]]; // Next empty singleton index for i-th cluster

				md_copy_block(DIMS, pos_dst, dst_dims, dst, singleton_dims, singleton, CFL_SIZE);

				idx[(long)labels[i]]++;

				// Debug output
				if (i % (long)(0.1 * N) == 0)
					debug_printf(DP_DEBUG3, "Binning: %f%\n", i * 1. / N * 100);
			}

			md_free(idx);
			md_free(singleton);
		}

		break;

	case 'R': { // Reorder: Reorder elements from src according to label

			debug_printf(DP_INFO, "Reordering...\n");

			// Find dimension of interest
			long dim;
			int count = 0;

			for (int i = 0; i < (int)DIMS; i++) {

				if (labels_dims[i] > 1) {

					dim = i;
					count++;
				}
			}

			assert(count == 1);

			// Check labels and find maximum
			float max = 0;

			for (int i = 0; i < labels_dims[dim]; i++) {

				assert(creal(labels[i]) >= 0); // Only positive labels allowed!

				max = (creal(labels[i]) > max) ? creal(labels[i]) : max;
			}

			assert(src_dims[dim] > max); 


			// Output
			long reorder_dims[DIMS];
			md_copy_dims(DIMS, reorder_dims, src_dims);
			reorder_dims[dim] = labels_dims[dim];

			complex float* reorder = create_cfl(argv[3], DIMS, reorder_dims);

			long singleton_dims[DIMS];
			md_select_dims(DIMS, ~(1u << dim), singleton_dims, src_dims);

			complex float* singleton = md_alloc(DIMS, singleton_dims, CFL_SIZE);

			long pos[DIMS] = { 0 };

			for (int i = 0; i < labels_dims[dim]; i++) {

				pos[dim] = creal(labels[i]);
				md_copy_block(DIMS, pos, singleton_dims, singleton, src_dims, src, CFL_SIZE);

				pos[dim] = i;
				md_copy_block(DIMS, pos, reorder_dims, reorder, singleton_dims, singleton, CFL_SIZE);				
			}

			unmap_cfl(DIMS, reorder_dims, reorder);
			md_free(singleton);
		}

		break;

	} // end switch case

	unmap_cfl(DIMS, labels_dims, labels);
	unmap_cfl(DIMS, src_dims, src);

	exit(0);
}

