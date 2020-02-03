/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 */

#include <sys/types.h>
#include <stdint.h>
#include <complex.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>


#include "num/multind.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

#define P_SHIFT(a) ((a)<0 ? -(a) : 0) // Shift to positive value
#define P_DIMS 20u
#define LOCMAX 30
#define MIXMAX 3
#define AVERMAX 150
#define KZMAX 150



enum paradise_dims {
		P_MIX_DIM,	 	/*mixed sequence number*/
		P_DYN_DIM, 		/*dynamic scan number*/
		P_CARD_DIM,		/*cardiac phase number*/
		P_ECHO_DIM,		/*echo number*/
		P_LOCA_DIM,		/*location number*/
		P_CHAN_DIM,		/*synco channel number*/
		P_EXTR1_DIM,	/*extra attribute 1 (semantics depend on type of scan)*/
		P_EXTR2_DIM,	/*extra attribute 2 (semantics depend on type of scan)*/
		P_KY_DIM,		/*ky*/
		P_KZ_DIM,		/*kz*/
		P_NA_DIM,		/*For spectroscopy?*/ 
		P_AVER_DIM,		/*sequence number of this signal average*/
		P_SIGN_DIM,		/*sign of measurement gradient used for this data vector (1 = positive, -1 = negative)*/
		P_RF_DIM,		/*sequence number of this rf echo (only for TSE, TFE, GraSE)*/
		P_GRAD_DIM,		/*sequence number of this gradient echo (only for EPI/GraSE)*/
		P_ENC_DIM,		/*encoding time (only for EPI/GraSE)*/
		P_RTOP_DIM,		/*R-top offset in ms*/
		P_RR_DIM,		/*RR-interval length in ms*/
		P_SIZE_DIM,		/*data vector size   in number of bytes (1 complex element = 2 floats = 8 bytes)*/
		P_OFFSET_DIM,	/*data vector offset in number of bytes (first data vector starts at offset 0)*/
};

#define P_MIX_FLAG (1u << P_MIX_DIM)
#define P_DYN_FLAG (1u << P_DYN_DIM)
#define P_CARD_FLAG (1u << P_CARD_DIM)
#define P_ECHO_FLAG (1u << P_ECHO_DIM)
#define P_LOCA_FLAG (1u << P_LOCA_DIM)
#define P_CHAN_FLAG (1u << P_CHAN_DIM)
#define P_EXTR1_FLAG (1u << P_EXTR1_DIM)
#define P_EXTR2_FLAG (1u << P_EXTR2_DIM)
#define P_KY_FLAG (1u << P_KY_DIM)
#define P_KZ_FLAG (1u << P_KZ_DIM)
#define P_NA_FLAG (1u << P_NA_DIM)
#define P_SIGN_FLAG (1u << P_SIGN_DIM)
#define P_AVER_FLAG (1u << P_AVER_DIM)
#define P_RF_FLAG (1u << P_RF_DIM)
#define P_GRAD_FLAG (1u << P_GRAD_DIM)
#define P_ENC_FLAG (1u << P_ENC_DIM)
#define P_RTOP_FLAG (1u << P_RTOP_DIM)
#define P_RR_FLAG (1u << P_RR_DIM)
#define P_SIZE_FLAG (1u << P_SIZE_DIM)
#define P_OFFSET_FLAG (1u << P_OFFSET_DIM)



char P_START[] = "START OF DATA VECTOR INDEX";
char P_NOI[] = "NOI";
char P_STD[] = "STD"; // Accepted line token
char P_REJ[] = "REJ"; // Rejected line token


struct p_param {

	int nav;	// Option for intermediate navigator extraction
	long nav_count; // Number of navigator lines
	bool data_vector;
	bool cine;	// Option for CINE sequences
	char inner_loop; // 'P': spoke_loop(partition_loop), 'S': partition_loop(spoke_loop)
	unsigned int kz_count[LOCMAX][MIXMAX][AVERMAX]; // kz-encoding counter for each location and each signal
	unsigned int t_count[LOCMAX][MIXMAX][AVERMAX][KZMAX]; // frame counter for each location, each signal and each partition
	long pos[P_DIMS]; // buffer
	long paradise_bounds[P_DIMS][2]; // minimum and maximum values for paradise parameters
	
};


static void xread(int fd, void* buf, size_t size)
{
	if (size != (size_t)read(fd, buf, size))
		error("reading file");
}

// Move file position to pos bytes from BEGINNING of the file
static void xseek(int fd, off_t pos)
{
	if (-1 == lseek(fd, pos, SEEK_SET))
			error("seeking");
}

// Move file position to pos bytes from CURRENT file position
static void cseek(int fd, off_t pos)
{
    if (-1 == lseek(fd, pos, SEEK_CUR))
		error("seeking");
}

// Read all paradise positions (indices) from a line
static void read_paradise_pos(long paradise_dims[P_DIMS], char* line)
{

	char* chunk = strtok(line, " "); 
	chunk = strtok(NULL, " ");
	
	int dim = 0;
	while( chunk != NULL ) {
		
		paradise_dims[dim] = atol(chunk);
		chunk = strtok(NULL, " ");
		dim++;
		
	}
}

// Determines the number of coils from the first measured STD-ADC
static void calc_coils(char* line, long* n_coils, struct p_param* params) 
{

	long pos_prev[P_DIMS];
	md_copy_dims(P_DIMS, pos_prev, params->pos);
	 
	read_paradise_pos(params->pos, line);
	
	if (params->paradise_bounds[P_CHAN_DIM][1] <= params->pos[P_CHAN_DIM])
		params->paradise_bounds[P_CHAN_DIM][1] = params->pos[P_CHAN_DIM];
	else
		*n_coils = params->paradise_bounds[P_CHAN_DIM][1] + 1;
	
}

// Determines which loop goes first: partition or spoke
static void det_loop_structure(char* line, struct p_param* params) 
{

	long pos_prev[P_DIMS];
	md_copy_dims(P_DIMS, pos_prev, params->pos);
	 
	read_paradise_pos(params->pos, line);
	
	if (pos_prev[P_KZ_DIM] == params->pos[P_KZ_DIM])
		params->inner_loop = 'S'; // Spoke loop is inner loop
	else
		params->inner_loop = 'P'; // Partition loop is inner loop
	
}

// Update frames for cine data
static void cine_update(long pos_prev[P_DIMS], struct p_param* params)
{
			
		long loc = params->pos[P_LOCA_DIM];
		long mix = params->pos[P_MIX_DIM];
		long aver = params->pos[P_AVER_DIM];

		if (params->inner_loop == 'S') {

			long kz = params->pos[P_KZ_DIM];
			
			//FIXME P_CARD_FLAG? - maybe P_AVER_FLAG, P_KY_FLAG are missing
			if((md_check_equal_dims(P_DIMS, pos_prev, params->pos, P_KZ_FLAG) || !md_check_equal_dims(P_DIMS, pos_prev, params->pos, P_LOCA_FLAG|P_CARD_FLAG)) && params->pos[P_KY_DIM] != 0) // partition stays the same or mix or location changes
				params->t_count[loc][mix][aver][kz]++;
		
			
		} else if (params->inner_loop == 'P') {
				
			// This sets the correct t_count:
			// - if the current indices of 'params->pos' differ from the previous indices 'pos_prev' in the given flags, we need to increase the time-counter
			// - directly after the transition to a new location/average we want to prevent to increase the counter (to keep the t_cout = 0 element. Therefore we check  params->pos[P_KY_DIM] != 0
			if(!md_check_equal_dims(P_DIMS, pos_prev, params->pos, P_LOCA_FLAG|P_KY_FLAG|P_AVER_FLAG) && params->pos[P_KY_DIM] != 0) { 
				
				params->t_count[loc][mix][aver][0]++;
				params->kz_count[loc][mix][aver] = 0;
				
			} else
				params->kz_count[loc][mix][aver]++;
			
		} else
			error("Loop structure unknown!\n");
		
}


// Since the 'General information' of the .list file are not reliable, 
// we must determine the dimensions by analyzing the data vector.
// We therefore search for the minimum and maximum value of all paradise_dims 
// and eventually need more checking for special sequence types like CINE...
static void calc_bounds(char* line, struct p_param* params) 
{

	long pos_prev[P_DIMS];
	md_copy_dims(P_DIMS, pos_prev, params->pos); // Indices of previous ADC
	
	read_paradise_pos(params->pos, line);	// Actual ADC indices
	
	char mode = 'c';
		
	switch (mode)
	{
		case 'c': // Conventional case
		{ 
			
			for (unsigned int i = 0; i < P_DIMS; i++) {

				params->paradise_bounds[i][0] = ( params->paradise_bounds[i][0] > params->pos[i]) ? params->pos[i] : params->paradise_bounds[i][0]; // update minimum
				params->paradise_bounds[i][1] = ( params->paradise_bounds[i][1] < params->pos[i]) ? params->pos[i] :  params->paradise_bounds[i][1]; // update maximum
				
			}
			
			if (params->nav && params->pos[P_KZ_DIM] % params->nav == 0)
				params->nav_count += 1;

			if (params->cine)		
				cine_update(pos_prev, params);
			
			break;
		}
	
	}
			

}

// Transfer indices from P_DIMS array to DIMS array
static void transfer_idx(long pos[DIMS], const struct p_param* params)
{
	pos[PHS2_DIM] = params->pos[P_KZ_DIM] + P_SHIFT(params->paradise_bounds[P_KZ_DIM][0]); // also for SMS and SoS partitions
	pos[TE_DIM]   = params->pos[P_ECHO_DIM] + P_SHIFT(params->paradise_bounds[P_ECHO_DIM][0]);
	pos[AVG_DIM]  = params->pos[P_AVER_DIM] + P_SHIFT(params->paradise_bounds[P_AVER_DIM][0]);
	pos[SLICE_DIM]  = params->pos[P_LOCA_DIM] + P_SHIFT(params->paradise_bounds[P_LOCA_DIM][0]); // for conventional Multislice slices
	pos[LEVEL_DIM] = params->pos[P_MIX_DIM] + P_SHIFT(params->paradise_bounds[P_MIX_DIM][0]); // Mixed signals, e.g. multiple ADCS per TR, ...
	
	long loc = params->pos[P_LOCA_DIM];
	long mix = params->pos[P_MIX_DIM];
	long aver = params->pos[P_AVER_DIM];
	long kz = params->pos[P_KZ_DIM];	
	
	if (params->cine) {	
		
		if (params->inner_loop == 'P')
			kz = 0;
		
		pos[PHS1_DIM] = 0;
		pos[TIME_DIM] = params->t_count[loc][mix][aver][kz];

	} else {
		
		pos[PHS1_DIM] = params->pos[P_KY_DIM] + P_SHIFT(params->paradise_bounds[P_KY_DIM][0]);
		pos[TIME_DIM] = params->pos[P_CARD_DIM] + P_SHIFT(params->paradise_bounds[P_CARD_DIM][0]);
		
	}
	
}

// Transfer indices from P_DIMS array to CFL array
static void transfer_idx_cfl(complex float* idx_singleton, const struct p_param* params)
{
	for (unsigned int i = 0; i < P_DIMS; i++)
		if (i != P_SIGN_DIM)
			idx_singleton[i] = 1. * (float)(params->pos[i] + P_SHIFT(params->paradise_bounds[i][0])) + 0i;
		else
			idx_singleton[i] = 1. * (float)params->pos[i] + 0i;
		
	long loc = params->pos[P_LOCA_DIM];
	long mix = params->pos[P_MIX_DIM];
	long aver = params->pos[P_AVER_DIM];
	long kz = params->pos[P_KZ_DIM];		
	
	
	if (params->cine) {
		
		if (params->inner_loop == 'P')
			kz = 0;
		
		idx_singleton[P_DYN_DIM] = 1.* (float)params->t_count[loc][mix][aver][kz] + 0i;
		
	}
		
}

// Reset values 
static void paradise_reset(FILE* fp, struct p_param* params)
{
	rewind(fp);
	params->data_vector = false;

	
	for (unsigned int i = 0; i < P_DIMS; i ++)
		params->pos[i] = 0;
	
	
	for (unsigned int i = 0; i < LOCMAX; i++) {
		for (unsigned int j = 0; j < MIXMAX; j++) {
			for (unsigned int k = 0; k < AVERMAX; k++) {
				for (unsigned int l = 0; l < KZMAX; l++) {
					params->t_count[i][j][k][l] = 0;
					params->kz_count[i][j][k] = 0;
				}
			}
		}
	}
	
	params->nav_count = 0;
		
}



static void philips_adc_read(int ifd, char* line, long pos[DIMS], const long dims[DIMS], complex float* out, const long idx_dims[DIMS], complex float* idx, long nav_dims[DIMS], complex float* nav, const long adc_dims[DIMS], complex float* buf, struct p_param* params) 
{

	long pos_prev[P_DIMS];
	md_copy_dims(P_DIMS, pos_prev, params->pos); // Indices of previous ADC
		
	read_paradise_pos(params->pos, line);

	xread(ifd, buf, dims[COIL_DIM] * params->pos[P_SIZE_DIM]);
	
	char mode = 'c';
	
	if (params->nav && params->pos[P_KZ_DIM] % params->nav == 0) // Navigator line
		mode = 'n';
		
	switch (mode)
	{
		case 'c': // Conventional case
		{
			if (params->cine)		
				cine_update(pos_prev, params);
			transfer_idx(pos, params);
			
			// Account for mixed signals with different read-out length by zero-padding
			if (params->pos[P_SIZE_DIM] < params->paradise_bounds[P_SIZE_DIM][1]) {
				
				long adc1_dims[DIMS];
				md_copy_dims(DIMS, adc1_dims, adc_dims);
				adc1_dims[READ_DIM] = (long)(params->pos[P_SIZE_DIM] * 1. / 8.); 
				complex float* buf_zeropad = md_alloc(DIMS, adc_dims, CFL_SIZE);
				
				md_resize(DIMS, adc_dims, buf_zeropad, adc1_dims, buf, CFL_SIZE);
				
				buf = buf_zeropad;
				md_free(buf_zeropad);
				
			}
				
			md_copy_block(DIMS, pos, dims, out, adc_dims, buf, CFL_SIZE); 
			
			
			if (idx != NULL) {

				long idx_singleton_dims[DIMS];
				md_select_dims(DIMS, TIME2_FLAG, idx_singleton_dims, idx_dims);
				
				complex float* idx_singleton = md_alloc(DIMS, idx_singleton_dims, CFL_SIZE);
				md_clear(DIMS, idx_singleton_dims, idx_singleton, CFL_SIZE);
				
				transfer_idx_cfl(idx_singleton, params);

				md_copy_block(DIMS, pos, idx_dims, idx, idx_singleton_dims, idx_singleton, CFL_SIZE);
							
			}
			
			break;
			
		}
		
		case 'n': // Navigator line
		{
			
			long pos1[DIMS] = { 0 };
			pos1[TIME_DIM] = params->nav_count;
			params->nav_count += 1;

			md_copy_block(DIMS, pos1, nav_dims, nav, adc_dims, buf, CFL_SIZE); 
			
			if (params->cine)		
				cine_update(pos_prev, params);
			
			break;
			
		}
		
	}
		
}


// Advance file position for rejected ADC
static void philips_adc_skip(int ifd, char* line, const int n_coils) 
{

	long pos[P_DIMS];
	read_paradise_pos(pos, line);
	
	cseek(ifd, pos[P_SIZE_DIM] * n_coils);

}

static const char usage_str[] = "<filename> <output> [<idx>]";
//	fprintf(fd, "Usage: %s [...] [-a A] <dat file> <output>\n", name);

static const char help_str[] = "Read data from Philips Paradise (.data/.list) files.";


int main_paradiseread(int argc, char* argv[argc])
{
	long dims[DIMS];
	md_singleton_dims(DIMS, dims);
	
	struct p_param params = { 
		0,
		0,
		false,
		false,
		0,
		{{{ 0 }}},
		{{{{ 0 }}}},
		{ 0 },
		{{ 0 }},
	};
	
	struct opt_s opts[] = {
		
		OPT_SET('c', &params.cine, "CINE"),		
		OPT_INT('n', &params.nav, "#", "Extract Navigator"),				

	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);
	
	
    // Create filenames
    int filename_length = strlen(argv[1]) + 5; // add 5 characters (.list/.data)

    char _data[] = ".data";
    char _list[] = ".list";
    
    char name_data[filename_length];
    char name_list[filename_length];

    
    strcpy(name_data, argv[1]);
    strcat(name_data, _data);

    strcpy(name_list, argv[1]);
    strcat(name_list, _list);

    // Read list-file line-by-line
    FILE* fp;
    char* line = NULL;
    size_t len = 0;

    fp = fopen(name_list, "r");
	
    if (fp == NULL) 
		error("List-file!");
	

	//--- Determine bounds ---
	long n_coils = 0;
	
	//// First, determine number of coils only
    while ((-1 != getline(&line, &len, fp)) && !n_coils) { // Iterate through all lines
		
		if (strstr(line, P_START)) // Skip general information
			params.data_vector = true;
		
		if (params.data_vector && strstr(line, P_STD))
			calc_coils(line, &n_coils, &params);
		
	}
		
	paradise_reset(fp, &params);
	
	//// Check if partition loop inside or outside of spoke loop
	if (params.cine) {

		debug_printf(DP_INFO, "CINE...\n");
		
		while ((-1 != getline(&line, &len, fp)) && !params.inner_loop) { // Iterate through all lines
			
			if (strstr(line, P_START)) // Skip general information
				params.data_vector = true;
			
			if (params.data_vector && strstr(line, P_STD))
				det_loop_structure(line, &params);
		
		}
		
		if (params.inner_loop == 'P')
			debug_printf(DP_INFO, "... inner loop is 'Partitions loop'!\n");
		
		else if (params.inner_loop == 'S')
			debug_printf(DP_INFO, "... inner loop is 'Spoke loop'!\n");
		
		else
			error("Cannot determine loop structure for CINE imaging!");
		
		paradise_reset(fp, &params);
		
	}
	

	
	//// Bounds
	int adcs = 0; 	// Total number of ADCs
	int line_count = 0; 		
    while ((-1 != getline(&line, &len, fp))) { // Iterate through all lines
		
		if (strstr(line, P_START)) { // Skip general information
			params.data_vector = true;
			// skip some more lines
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = 0;
		}
	
		if (params.data_vector && (line_count % n_coils) == 0 && strstr(line, P_STD)) {
			
			calc_bounds(line, &params);
			adcs++;
			
		}
		
		line_count++;
		
	}

	//// Assign relevant values
	dims[READ_DIM] = (long)(params.paradise_bounds[P_SIZE_DIM][1] * 1. / 8.); 
	dims[COIL_DIM] = (long)(labs(params.paradise_bounds[P_CHAN_DIM][0]) + labs(params.paradise_bounds[P_CHAN_DIM][1]) +  1);
	dims[LEVEL_DIM] = (long)(labs(params.paradise_bounds[P_MIX_DIM][0]) + labs(params.paradise_bounds[P_MIX_DIM][1]) +  1);
	dims[SLICE_DIM] = (long)(labs(params.paradise_bounds[P_LOCA_DIM][0]) + labs(params.paradise_bounds[P_LOCA_DIM][1]) +  1);
	dims[PHS2_DIM] = (long)(labs(params.paradise_bounds[P_KZ_DIM][0]) + labs(params.paradise_bounds[P_KZ_DIM][1]) +  1);
	dims[AVG_DIM] = (long)(labs(params.paradise_bounds[P_AVER_DIM][0]) + labs(params.paradise_bounds[P_AVER_DIM][1]) +  1);


	if (params.cine) {
		
		dims[PHS1_DIM] = 1;
		
		// find maximum
		for (unsigned int i = 0; i < LOCMAX; i++)
			for (unsigned int j = 0; j < MIXMAX; j++) 
				for (unsigned int k = 0; k < AVERMAX; k++) 
					for (unsigned int l = 0; l < KZMAX; l++) 				
						dims[TIME_DIM] = (params.t_count[i][j][k][l] > dims[TIME_DIM]) ? params.t_count[i][j][k][l] : dims[TIME_DIM];
	
 		dims[TIME_DIM] += 1; // Account for zero-based indexing
					
	} else {
		
		dims[PHS1_DIM] = (long)(labs(params.paradise_bounds[P_KY_DIM][0]) + labs(params.paradise_bounds[P_KY_DIM][1]) +  1);
		dims[TIME_DIM] = (long)(labs(params.paradise_bounds[TIME_DIM][0]) + labs(params.paradise_bounds[TIME_DIM][1]) +  1);
		
	}
	
	debug_print_dims(DP_DEBUG3, DIMS, dims);
	
	long n_nav = params.nav_count; // Number of navigator lines
	
	paradise_reset(fp, &params);

	//--- Determine start offset ---
	off_t start = 0;
	while (start == 0 && (-1 != getline(&line, &len, fp))) { // Iterate through all lines
		
		if (strstr(line, P_START)) // Skip general information
			params.data_vector = true;
		
		if (params.data_vector && (strstr(line, P_STD) || strstr(line, P_REJ))) {
			
			long pos[P_DIMS];
			read_paradise_pos(pos, line);
			start = pos[P_OFFSET_DIM];
			
		}
	}

	paradise_reset(fp, &params);


	//--- Read the data ---
	long idx_dims[DIMS];
	md_select_dims(DIMS, PHS1_FLAG|PHS2_FLAG|TIME_FLAG|LEVEL_FLAG|SLICE_FLAG|AVG_FLAG, idx_dims, dims);
	idx_dims[TIME2_DIM] = P_DIMS;
	
	complex float* idx = NULL;
	if (argc == 4)
		idx = create_cfl(argv[3], DIMS, idx_dims);
	
	complex float* out = create_cfl(argv[2], DIMS, dims);
	md_clear(DIMS, dims, out, CFL_SIZE);

	long adc_dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|COIL_FLAG, adc_dims, dims);
	void* buf = md_alloc(DIMS, adc_dims, CFL_SIZE);
	
	// Navigator
	long nav_dims[DIMS] = { 0 };
	complex float* nav = NULL;
	
	if (params.nav) {
		
		// Generate name
		char nav_postfix[] = "_nav";
		long name_length = strlen(argv[2]) + 4; // "_nav"
		char name_nav[name_length];
		strcpy(name_nav, argv[2]);
		strcat(name_nav, nav_postfix);
		
		md_copy_dims(DIMS, nav_dims, adc_dims);
		nav_dims[TIME_DIM] = n_nav;
		
		nav = create_cfl(name_nav, DIMS, nav_dims);

	}
	

	int ifd;
    if (-1 == (ifd = open(name_data, O_RDONLY)))
		error("error opening file.");
	
	xseek(ifd, start);
		
	long pos[DIMS] = { 0 };
	
	line_count = 0;
	while ((-1 != getline(&line, &len, fp))) { // Iterate through all lines
		
		if (strstr(line, P_START)) { // Skip general information
			params.data_vector = true;
			// skip some more lines
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = getline(&line, &len, fp);
			line_count = 0;
		}
	
		if (params.data_vector && (line_count % n_coils) == 0 && strstr(line, P_STD)) { 
			philips_adc_read(ifd, line, pos, dims, out, idx_dims, idx, nav_dims, nav, adc_dims, buf,  &params);
			adcs--;
				
		} else if (params.data_vector && (line_count % n_coils) == 0 && strstr(line, P_REJ)) { 
			
			philips_adc_skip(ifd, line, n_coils);

		}

		line_count++;

	}
	
	if (adcs != 0)
		error("Inconsistent number of ADCs!");
	
	
	fclose(fp);
    if (line)
		free(line);

	unmap_cfl(DIMS, dims, out);
	
	if (argc == 4)
		unmap_cfl(DIMS, idx_dims, idx);
	
	if (params.nav)
		unmap_cfl(DIMS, nav_dims, nav);

	return 0;
}


