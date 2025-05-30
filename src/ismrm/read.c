/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012, 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <complex.h>

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "ismrm/xml_wrapper.h"

#include "read.h"

// constexpr
#define ismrmrd_default_limit_macro (struct limit_s){ .size = 1,\
					 .size_hdr = 1,\
					 .max_hdr = -1,\
					 .min_hdr = -1,\
					 .max_idx = -1,\
					 .min_idx = -1,\
					 .center = -1 }

const struct limit_s ismrmrd_default_limit = ismrmrd_default_limit_macro;

// FIXME: does not deal correctly with repetitions (and others stuff)

struct isrmrm_config_s ismrm_default_config = {

	.idx_encoding = 0,

	.dim_mapping = {

		[ ISMRMRD_READ_DIM ]		= READ_DIM,
		[ ISMRMRD_COIL_DIM ]		= COIL_DIM,
		[ ISMRMRD_PHS1_DIM ]		= PHS1_DIM,
		[ ISMRMRD_PHS2_DIM ]		= PHS2_DIM,
		[ ISMRMRD_AVERAGE_DIM ] 	= AVG_DIM,
		[ ISMRMRD_SLICE_DIM ]		= SLICE_DIM,
		[ ISMRMRD_CONTRAST_DIM ]	= -1,
		[ ISMRMRD_PHASE_DIM ]		= -1,
		[ ISMRMRD_REPETITION_DIM ]	= -1,
		[ ISMRMRD_SET_DIM ]		= -1,
		[ ISMRMRD_SEGMENT_DIM ]		= -1,
		[ ISMRMRD_NAMED_DIMS ... ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS - 1] = -1,
	},

	.limits = { [0 ... ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS - 1]
			= ismrmrd_default_limit_macro
		  },

	.slice_ord = ISMRMRD_SLICE_ASCENDING,

	.check_dims_with_acquisition = true,

	.merge_dims = 0,
	.shift = MD_BIT(1) | MD_BIT(2) | MD_BIT(3),

	.overwriting_idx = -1,
	.measurement = -1,
	.repetition = -1,

	.ismrm_cpp_state = NULL,
};




static const char* ismrmrd_get_dim_string(enum ISMRMRD_mri_dims code)
{
	switch (code) {

		case ISMRMRD_READ_DIM: return "ISMRMRD_READ_DIM";
		case ISMRMRD_COIL_DIM: return "ISMRMRD_COIL_DIM";
		case ISMRMRD_PHS1_DIM: return "ISMRMRD_PHS1_DIM";
		case ISMRMRD_PHS2_DIM: return "ISMRMRD_PHS2_DIM";
		case ISMRMRD_AVERAGE_DIM: return "ISMRMRD_AVERAGE_DIM";
		case ISMRMRD_SLICE_DIM: return "ISMRMRD_SLICE_DIM";
		case ISMRMRD_CONTRAST_DIM: return "ISMRMRD_CONTRAST_DIM";
		case ISMRMRD_PHASE_DIM: return "ISMRMRD_PHASE_DIM";
		case ISMRMRD_REPETITION_DIM: return "ISMRMRD_REPETITION_DIM";
		case ISMRMRD_SET_DIM: return "ISMRMRD_SET_DIM";
		case ISMRMRD_SEGMENT_DIM: return "ISMRMRD_SEGMENT_DIM";
		case ISMRMRD_NAMED_DIMS: break;
		default: break;
	}

	switch (code - ISMRMRD_NAMED_DIMS) {

		case 0: return "ISMRMRD_USER0_DIM"; 
		case 1: return "ISMRMRD_USER1_DIM"; 
		case 2: return "ISMRMRD_USER2_DIM"; 
		case 3: return "ISMRMRD_USER3_DIM"; 
		case 4: return "ISMRMRD_USER4_DIM"; 
		case 5: return "ISMRMRD_USER5_DIM"; 
		case 6: return "ISMRMRD_USER6_DIM"; 
		case 7: return "ISMRMRD_USER7_DIM"; 
		case 8: return "ISMRMRD_USER8_DIM"; 
	}

	assert(code < ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS);

	return "ISMRMRD_UNKNOWN_DIM";
}

static int ismrmrd_get_idx(enum ISMRMRD_mri_dims code, struct ISMRMRD_EncodingCounters* idx)
{
	assert(code < ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS);

	switch (code) {

		case ISMRMRD_READ_DIM: return -1;
		case ISMRMRD_COIL_DIM: return -1;
		case ISMRMRD_PHS1_DIM: return idx->kspace_encode_step_1;
		case ISMRMRD_PHS2_DIM: return idx->kspace_encode_step_2;
		case ISMRMRD_AVERAGE_DIM: return idx->average;
		case ISMRMRD_SLICE_DIM: return idx->slice;
		case ISMRMRD_CONTRAST_DIM: return idx->contrast;
		case ISMRMRD_PHASE_DIM: return idx->phase;
		case ISMRMRD_REPETITION_DIM: return idx->repetition;
		case ISMRMRD_SET_DIM: return idx->set;
		case ISMRMRD_SEGMENT_DIM: return idx->segment;
		case ISMRMRD_NAMED_DIMS: break;
		default: break;
	}

	return idx->user[code - ISMRMRD_NAMED_DIMS];
}



static void debug_print_ISMRMRD_index(int level, struct ISMRMRD_EncodingCounters idx)
{
	for (enum ISMRMRD_mri_dims i = 0; i < ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS; i++)
		debug_printf(level, "%s: %d\n", ismrmrd_get_dim_string(i), ismrmrd_get_idx(i, &idx));
}

static void debug_print_ISMRMRD_acq(int level, struct ISMRMRD_AcquisitionHeader head)
{
	debug_printf(level, "%s: %u\n", "version", head.version);
	debug_printf(level, "%s: %lu\n", "flags", head.flags);
	debug_printf(level, "%s: %u\n", "measurement_uid", head.measurement_uid);
	debug_printf(level, "%s: %u\n", "scan_counter", head.scan_counter);
	debug_printf(level, "%s: %u\n", "acquisition_time_stamp", head.acquisition_time_stamp);
	debug_printf(level, "%s: %u, %u, %u\n", "physiology_time_stamp", head.physiology_time_stamp[0], head.physiology_time_stamp[1], head.physiology_time_stamp[2]);
	debug_printf(level, "%s: %u\n", "number_of_samples", head.number_of_samples);
	debug_printf(level, "%s: %u\n", "available_channels", head.available_channels);
	debug_printf(level, "%s: %u\n", "active_channels", head.active_channels);
	debug_printf(level, "%s: %u\n", "discard_pre", head.discard_pre);
	debug_printf(level, "%s: %u\n", "discard_post", head.discard_post);
	debug_printf(level, "%s: %u\n", "center_sample", head.center_sample);
	debug_printf(level, "%s: %u\n", "encoding_space_ref", head.encoding_space_ref);
	debug_printf(level, "%s: %u\n", "trajectory_dimensions", head.trajectory_dimensions);

	debug_print_ISMRMRD_index(level, head.idx);
}

static void ismrm_conf_merge_unmapped_dims(struct isrmrm_config_s* config);
static void ismrm_conf_to_dims(const struct isrmrm_config_s* config, int N, long dims[__VLA(N)]);
static bool ismrm_validate_limits(struct isrmrm_config_s* config);

void ismrm_read_dims(const char* datafile, struct isrmrm_config_s* config, int N, long dims[N])
{
	ismrm_read_encoding_limits(datafile, config);

	md_singleton_dims(N, dims);

	//cross check indices
	if (config->check_dims_with_acquisition) {

		ismrm_read(datafile, config, N, dims, NULL);
		ismrm_validate_limits(config);
	}

	ismrm_conf_merge_unmapped_dims(config);
	ismrm_conf_to_dims(config, N, dims);
}

void ismrm_stream_read_dims(struct isrmrm_config_s* config, int N, long dims[N])
{
	assert(NULL != config->ismrm_cpp_state);

	md_singleton_dims(N, dims);
	ismrm_stream_read_meta(config);
	ismrm_conf_merge_unmapped_dims(config);
	ismrm_conf_to_dims(config, N, dims);
}

static bool ismrm_validate_limits(struct isrmrm_config_s* config)
{
	bool changed = false;

	for (enum ISMRMRD_mri_dims i = ISMRMRD_PHS1_DIM; i < ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS; i++) {

		if ((1 != config->limits[i].size) && (config->limits[i].max_idx == config->limits[i].min_idx)) {

			debug_printf(DP_WARN, "Dimension \"%s\" has size %ld but all acquisitions have the same index (%ld)!\n      => Set dimension to one!\n",
						ismrmrd_get_dim_string(i), config->limits[i].size, config->limits[i].max_idx);

			config->limits[i].size = 1;
			config->merge_dims = MD_SET(config->merge_dims, i);
			changed = true;
		}

		if ((1 == config->limits[i].size) && (config->limits[i].max_idx == config->limits[i].min_idx) && (0 < config->limits[i].max_idx)) {

			debug_printf(DP_WARN, "Dimension \"%s\" has size %ld but all acquisitions have the same index (%ld)!\n      => Set indices to 0!\n",
						ismrmrd_get_dim_string(i), config->limits[i].size, config->limits[i].max_idx);

			config->limits[i].size = 1;
			config->merge_dims = MD_SET(config->merge_dims, i);
			changed = true;
		}

		if ((1 == config->limits[i].size) && (config->limits[i].max_idx > config->limits[i].min_idx)) {

			debug_printf(DP_WARN, "Dimension \"%s\" has size %ld but acquisitions extend from %ld to %ld!\n      => All indices are set to 0, check for overwriting data!\n",
						ismrmrd_get_dim_string(i), config->limits[i].size, config->limits[i].min_idx, config->limits[i].max_idx);

			config->limits[i].size = 1;
			config->merge_dims = MD_SET(config->merge_dims, i);
			changed = true;
		}

		if ((!MD_IS_SET(config->shift, i)) && (config->limits[i].max_idx > config->limits[i].min_idx) && (config->limits[i].max_idx + 1 < config->limits[i].size)) {

			debug_printf(DP_WARN, "Dimension \"%s\" has size %ld but acquisitions extend only to %ld!\n      => Dimension is reduced!\n",
						ismrmrd_get_dim_string(i), config->limits[i].size, config->limits[i].max_idx);

			config->limits[i].size = config->limits[i].max_idx + 1;
			changed = true;
		}
	}
	return changed;
}

static void ismrm_conf_merge_unmapped_dims(struct isrmrm_config_s* config)
{
	for (enum ISMRMRD_mri_dims i = ISMRMRD_PHS1_DIM; i < ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS; i++) {

		if ((1 != config->limits[i].size) && (-1 == config->dim_mapping[i])) {

			debug_printf(DP_WARN, "Dimension \"%s\" has size %ld but is not mapped to BART dimension!\n      => All indices are set to 0, check for overwriting data!\n",
						ismrmrd_get_dim_string(i), config->limits[i].size);

			config->limits[i].size = 1;
			config->merge_dims = MD_SET(config->merge_dims, i);
		}
	}
}

static void ismrm_conf_to_dims(const struct isrmrm_config_s* config, int N, long dims[N])
{
	for (int i = ISMRMRD_PHS1_DIM; i < ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS; i++)
		assert ((1 == config->limits[i].size) || (-1 != config->dim_mapping[i]));

	long max[N];
	long min[N];
	long ctr[N];

	for (int i = 0; i < N; i++) {

		max[i] = -1;
		min[i] = -1;
		ctr[i] = -1;
	}

	for (int i = 0; i < (int)ARRAY_SIZE(config->limits); i++) {

		if (1 == config->limits[i].size)
			continue;

		if (-1 == config->dim_mapping[i])
			error("ISMRMRD dimension %d is not mapped to BART dimension, but dimension is not 1!\n", i);

		assert(config->dim_mapping[i] < N);

		dims[config->dim_mapping[i]] = config->limits[i].size;

		max[config->dim_mapping[i]] = config->limits[i].max_hdr;
		min[config->dim_mapping[i]] = config->limits[i].min_hdr;
		ctr[config->dim_mapping[i]] = config->limits[i].center;
	}

	debug_printf(DP_DEBUG1, "Dims:   "); debug_print_dims(DP_DEBUG1, N, dims);
	debug_printf(DP_DEBUG1, "Max:    "); debug_print_dims(DP_DEBUG1, N, max);
	debug_printf(DP_DEBUG1, "Min:    "); debug_print_dims(DP_DEBUG1, N, min);
	debug_printf(DP_DEBUG1, "Center: "); debug_print_dims(DP_DEBUG1, N, ctr);
}

inline static bool set_pos(struct isrmrm_config_s* config, int N, long pos[N], int map, int idx)
{
	assert(config->dim_mapping[map] < N);

	if (-1 == config->limits[map].min_idx)
		config->limits[map].min_idx = idx;
	else
		config->limits[map].min_idx = MIN(config->limits[map].min_idx, idx);

	if (-1 == config->limits[map].max_idx)
		config->limits[map].max_idx = idx;
	else
		config->limits[map].max_idx = MAX(config->limits[map].max_idx, idx);


	if ((1 == config->limits[map].size) && MD_IS_SET(config->merge_dims, map))
		idx = 0;

	if (ISMRMRD_SLICE_DIM == map) {

		switch (config->slice_ord) {

			case ISMRMRD_SLICE_ASCENDING: break;
			case ISMRMRD_SLICE_INTERLEAVED:
			{
				long max = config->limits[map].size;

				if (0 == max % 2)
					idx = (idx < max / 2) ? idx * 2 : 2 * idx - max + 1;
				else
					idx = (idx <= max / 2) ? idx * 2 : 2 * (idx - max / 2) - 1;
			}
			break;
			case ISMRMRD_SLICE_INTERLEAVED_SIEMENS:
			{
				long max = config->limits[map].size;

				if (0 == max % 2)
					idx = (idx < max / 2) ? idx * 2 + 1 : 2 * idx - max;
				else
					idx = (idx <= max / 2) ? idx * 2 : 2 * (idx - max / 2) - 1;
			}
			break;
		}
	}

	if ((0 > idx) || (idx >= config->limits[map].size)) {

		static bool warn = true;

		if (warn) {

			warn = false;
			debug_printf(DP_WARN, "Acquisition index %d out of bounds for \"%s\" (size=%ld)! -> set to 0.\n"
						"Check for overwriting.\n"
						"Further warnings will be suppressed!\n",
					idx, ismrmrd_get_dim_string((unsigned)map), config->limits[map].size);
		}

		idx = 0;
	}

	if (MD_IS_SET(config->shift, map) && (-1 != config->limits[map].center))
		idx += (config->limits[map].size / 2 - config->limits[map].center);

	if (0 > config->dim_mapping[map]) {

		if (0 == idx)
			return true;

		error("BART ISMRMRD read: Non-trivial idx for unmapped dimension.\n");
	}

	assert(0 <= config->dim_mapping[map]);

	pos[config->dim_mapping[map]] = idx;
	return true;
}

static bool ismrm_read_idx(struct isrmrm_config_s* config, struct ISMRMRD_EncodingCounters idx, int N, long pos[N])
{
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	bool result = true;

	for (int i = ISMRMRD_PHS1_DIM; i < ISMRMRD_NAMED_DIMS + ISMRMRD_USER_INTS; i++)
		result = result && set_pos(config, N, pos, i, ismrmrd_get_idx((unsigned)i, &idx));

	return result;
}

static bool ismrmrd_convert_acquisition(struct isrmrm_config_s* config, const ISMRMRD_Acquisition* acq, int N, const long dims[N], long strs[N], long pos[N], complex float* buf);

void ismrm_read(const char* datafile, struct isrmrm_config_s* config, int N, long dims[N], complex float* buf)
{
	ISMRMRD_Dataset d;
	ismrmrd_init_dataset(&d, datafile, "/dataset");
	ismrmrd_open_dataset(&d, false);

	long number_of_acquisitions = ismrmrd_get_number_of_acquisitions(&d);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	ISMRMRD_Acquisition acq;

	config->convert_state = (struct ismrmrd_convert_state){ 0 };

	for (long i = 0; i < number_of_acquisitions; i++) {

		ismrmrd_init_acquisition(&acq);
		ismrmrd_read_acquisition(&d, i, &acq);
		if ((!ismrmrd_convert_acquisition(config, &acq, N, dims, strs, pos, buf)) && (NULL != buf))
			debug_printf(DP_WARN, "SKIPPED ACQUISITION %ld!\n", i);
	}

	debug_printf(DP_DEBUG2, "Counter flags: ");
	debug_print_dims(DP_DEBUG2, 64, config->convert_state.counter_flags);

	if (NULL != buf)
		debug_printf(DP_DEBUG1, "In total %ld acquisitions copied!\n", config->convert_state.counter);
}


static bool ismrmrd_convert_acquisition(struct isrmrm_config_s* config, const ISMRMRD_Acquisition* acq, int N, const long dims[N], long strs[N], long pos[N], complex float* buf)
{
	config->convert_state.attempts++;
	bool skip = false;

	if ((-1 != config->measurement) && (config->convert_state.counter_flags[ISMRMRD_ACQ_LAST_IN_MEASUREMENT - 1] != config->measurement))
		skip = true;

	for (int j = 0; j < (int)ARRAY_SIZE(config->convert_state.counter_flags); j++)
		config->convert_state.counter_flags[j] += MD_IS_SET(acq->head.flags, j) ? 1 : 0;

	if (skip)
		return false;


	if (MD_IS_SET(acq->head.flags, (ISMRMRD_ACQ_IS_NOISE_MEASUREMENT - 1)))
		return false;

	if (MD_IS_SET(acq->head.flags, (ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION - 1)))
		return false;

	if (MD_IS_SET(acq->head.flags, (ISMRMRD_ACQ_IS_NAVIGATION_DATA - 1)))
		return false;

	if (MD_IS_SET(acq->head.flags, (ISMRMRD_ACQ_IS_PHASECORR_DATA - 1)))
		return false;

	if (MD_IS_SET(acq->head.flags, (ISMRMRD_ACQ_IS_REVERSE  - 1))) {

		static bool warn = true;

		if (warn && (NULL != buf)) {

			warn = false;
			debug_printf(DP_WARN, "Acquisition is reverse! This is probably not handled correctly! Further warnings will be suppressed!\n");
		}
	}

	if (acq->head.encoding_space_ref != config->idx_encoding)
		return false;

	if ((!ismrm_read_idx(config, acq->head.idx, N, pos)) || (NULL == buf))
		return false;

	long channels = acq->head.available_channels;
	if (acq->head.available_channels != acq->head.active_channels)
		error("All channels must be active, but (%d/%d) are active!\n", acq->head.active_channels, acq->head.available_channels);

	long samples = acq->head.number_of_samples;

	assert(channels == dims[config->dim_mapping[ISMRMRD_COIL_DIM]]);
	assert(samples + acq->head.discard_post + acq->head.discard_pre == dims[config->dim_mapping[ISMRMRD_READ_DIM]]);

	long adc_dims[N];
	long adc_strs[N];

	md_singleton_dims(N, adc_dims);
	assert(config->dim_mapping[ISMRMRD_READ_DIM] < config->dim_mapping[ISMRMRD_COIL_DIM]);
	adc_dims[config->dim_mapping[ISMRMRD_READ_DIM]] = samples;
	adc_dims[config->dim_mapping[ISMRMRD_COIL_DIM]] = channels;

	md_calc_strides(N, adc_strs, adc_dims, CFL_SIZE);

	if (0 != md_znorm2(N, adc_dims, strs, &MD_ACCESS(N, strs, pos, buf))) {

		static bool warn = true;

		if  (-1 != config->overwriting_idx)
			warn = false;

		if (warn) {

			warn = false;
			debug_printf(DP_WARN, "Acquisition would overwrite data! -> Skipped\n      Further warnings will be suppressed!\n");
		}

		if (config->convert_state.overwrite_counter < config->overwriting_idx) {

			md_clear(N, dims, buf, CFL_SIZE);
			config->convert_state.overwrite_counter++;
		} else {

			return false;
		}
	}

	debug_print_ISMRMRD_acq(DP_DEBUG3, acq->head);
	debug_print_dims(DP_DEBUG3, N, pos);
	debug_print_dims(DP_DEBUG3, N, dims);
	debug_print_dims(DP_DEBUG3, N, strs);
	debug_print_dims(DP_DEBUG3, N, adc_dims);
	debug_print_dims(DP_DEBUG3, N, adc_strs);

	md_copy_block2(N, pos, dims, strs, buf, adc_dims, adc_strs, acq->data, CFL_SIZE);

	debug_printf(DP_DEBUG3, "Copied %u %u\n", acq->head.measurement_uid, acq->head.scan_counter);

	config->convert_state.counter++;

	return true;
}

long ismrm_stream_read(struct isrmrm_config_s* conf, int N, const long dims[N], long pos[N], complex float* out)
{
	assert(NULL != conf->ismrm_cpp_state);

	ISMRMRD_Acquisition acq;
	ismrmrd_init_acquisition(&acq);

	long bytes = ismrm_stream_read_acquisition(conf, &acq);
	if (0 == bytes)
		return bytes;

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	if(!ismrmrd_convert_acquisition(conf, &acq, N, dims, strs, pos, out))
		debug_printf(DP_WARN, "SKIPPED ACQUISITION %ld!\n", conf->convert_state.attempts);

	return bytes;
}



void ismrm_print_xml(const char* filename)
{
	ISMRMRD_Dataset d;
	ismrmrd_init_dataset(&d, filename, "/dataset");
	ismrmrd_open_dataset(&d, false);

	const char* xml =  ismrmrd_read_header(&d);

	bart_printf("%s", xml);

	xfree(xml);

	ismrmrd_close_dataset(&d);
}
