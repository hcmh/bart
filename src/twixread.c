/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <sys/types.h>
#include <stdint.h>
#include <complex.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


/* Information about twix files can be found here:
 * (Matlab code by Philipp Ehses and others, Yarra by Tobias Block)
 * https://github.com/cjohnevans/Gannet2.0/blob/master/mapVBVD.m
 * https://bitbucket.org/yarra-dev/yarramodules-setdcmtags/src/
 */ 
struct hdr_s {

	uint32_t offset;
	uint32_t nscans;
};

struct entry_s {

	uint32_t measid;
	uint32_t fileid;
	uint64_t offset;
	uint64_t length;
	char patient[64];
	char protocol[64];
};

static void xread(int fd, void* buf, size_t size)
{
	if (size != (size_t)read(fd, buf, size))
		error("reading file");
}

static void xseek(int fd, off_t pos)
{
	if (-1 == lseek(fd, pos, SEEK_SET))
		error("seeking");
}

static bool siemens_meas_setup(int fd, struct hdr_s* hdr)
{
	off_t start = 0;

	xseek(fd, start);
	xread(fd, hdr, sizeof(struct hdr_s));

	// check for VD version
	bool vd = ((0 == hdr->offset) && (hdr->nscans < 64));

	if (vd) {
	
		assert((0 < hdr->nscans) && (hdr->nscans < 30));

		struct entry_s entries[hdr->nscans];
		xread(fd, &entries, sizeof(entries));

		int n = hdr->nscans - 1;

		debug_printf(DP_INFO, "VD Header. MeasID: %d FileID: %d Scans: %d\n",
					entries[n].measid, entries[n].fileid, hdr->nscans);

		debug_printf(DP_INFO, "Patient: %.64s\nProtocol: %.64s\n", entries[n].patient, entries[n].protocol);


		start = entries[n].offset;

		// reread offset
		xseek(fd, start);
		xread(fd, &hdr->offset, sizeof(hdr->offset));

	} else {

		debug_printf(DP_INFO, "VB Header.\n");
		hdr->nscans = 1;
	}

	start += hdr->offset;

	xseek(fd, start);

	return vd;
}


struct mdh1 {

	uint32_t flags_dmalength;
	int32_t measUID;
	uint32_t scounter;
	uint32_t timestamp;
	uint32_t pmutime;
};

struct mdh2 {	// second part of mdh

	uint32_t evalinfo[2];
	uint16_t samples;
	uint16_t channels;
	uint16_t sLC[14];
	uint16_t dummy1[2];
	uint16_t clmnctr;
	uint16_t dummy2[5];
	uint16_t linectr;
	uint16_t partctr;
};


static int siemens_bounds(bool vd, int fd, long min[DIMS], long max[DIMS])
{
	char scan_hdr[vd ? 192 : 0];
	size_t size = sizeof(scan_hdr);

	if (size != (size_t)read(fd, scan_hdr, size))
		return -1;

	long pos[DIMS] = { 0 };

	for (pos[COIL_DIM] = 0; pos[COIL_DIM] < max[COIL_DIM]; pos[COIL_DIM]++) {

		char chan_hdr[vd ? 32 : 128];
		size_t size = sizeof(chan_hdr);

		if (size != (size_t)read(fd, chan_hdr, size))
			return -1;

		struct mdh2 mdh;
		memcpy(&mdh, vd ? (scan_hdr + 40) : (chan_hdr + 20), sizeof(mdh));

		if (0 == max[READ_DIM]) {

			max[READ_DIM] = mdh.samples;
			//max[COIL_DIM] = mdh.channels;
		}

		if ((mdh.evalinfo[0] & (1 << 5))) {

//			debug_printf(DP_WARN, "SYNC\n");

			struct mdh1 mdh1;
			memcpy(&mdh1, vd ? scan_hdr : chan_hdr, sizeof(mdh1));

			size_t dma_length = mdh1.flags_dmalength & 0x01FFFFFFL;
			size_t offset = sizeof(scan_hdr) + sizeof(chan_hdr);

			if (dma_length < offset)
				error("dma_length < offset.\n");

			if (-1 == lseek(fd, dma_length - offset, SEEK_CUR))
				error("seeking");

			return 0;

		} else if (max[READ_DIM] != mdh.samples) {

			return -1;

		} else {

			max[COIL_DIM] = mdh.channels;
		}

		if (max[COIL_DIM] != mdh.channels)
			return -1;

		pos[PHS1_DIM]	= mdh.sLC[0];
		pos[AVG_DIM]	= mdh.sLC[1];
		pos[SLICE_DIM]	= mdh.sLC[2];
		pos[PHS2_DIM]	= mdh.sLC[3];
		pos[TE_DIM]	= mdh.sLC[4];
		pos[COEFF_DIM]	= mdh.sLC[5];
		pos[TIME_DIM]	= mdh.sLC[6];
		pos[TIME2_DIM]	= mdh.sLC[7];


		for (unsigned int i = 0; i < DIMS; i++) {

			max[i] = MAX(max[i], pos[i] + 1);
			min[i] = MIN(min[i], pos[i] + 0);
		}

		size = mdh.samples * CFL_SIZE;
		char buf[size];

		if (size != (size_t)read(fd, buf, size))
			return -1;
	}

	return 0;
}


static int siemens_adc_read(bool vd, int fd, bool linectr, bool partctr, const long dims[DIMS], long pos[DIMS], complex float* buf, uint32_t* buf_pmu, uint32_t* buf_timestamp)
{
	char scan_hdr[vd ? 192 : 0];

	xread(fd, scan_hdr, sizeof(scan_hdr));

	struct mdh1 mdh_1;
	memcpy(&mdh_1, scan_hdr, sizeof(mdh_1));

	*buf_pmu = mdh_1.pmutime;
	*buf_timestamp = mdh_1.timestamp;
	//debug_printf(DP_DEBUG1, "PMUtimestamp: %lu\n", mdh_1.pmutime);


	for (pos[COIL_DIM] = 0; pos[COIL_DIM] < dims[COIL_DIM]; pos[COIL_DIM]++) {

		char chan_hdr[vd ? 32 : 128];
		xread(fd, chan_hdr, sizeof(chan_hdr));

		struct mdh2 mdh;
		memcpy(&mdh, vd ? (scan_hdr + 40) : (chan_hdr + 20), sizeof(mdh));

		if ((mdh.evalinfo[0] & (1 << 5))
			 || (dims[READ_DIM] != mdh.samples)) {

//			debug_printf(DP_WARN, "SYNC\n");

			struct mdh1 mdh1;
			memcpy(&mdh1, vd ? scan_hdr : chan_hdr, sizeof(mdh1));

			size_t dma_length = mdh1.flags_dmalength & 0x01FFFFFFL;
			size_t offset = sizeof(scan_hdr) + sizeof(chan_hdr);

			if (dma_length < offset)
				error("dma_length < offset.\n");

			if (-1 == lseek(fd, dma_length - offset, SEEK_CUR))
				error("seeking");

			return 0;
		}


		if (0 == pos[COIL_DIM]) {

			// TODO: rethink this
			pos[PHS1_DIM]	= mdh.sLC[0] + (linectr ? mdh.linectr : 0);
			pos[AVG_DIM]	= mdh.sLC[1];
			pos[SLICE_DIM]	= mdh.sLC[2];
			pos[PHS2_DIM]	= mdh.sLC[3] + (partctr ? mdh.partctr : 0);
			pos[TE_DIM]	= mdh.sLC[4];
			pos[COEFF_DIM]	= mdh.sLC[5];
			pos[TIME_DIM]	= mdh.sLC[6];
			pos[TIME2_DIM]	= mdh.sLC[7];
		}

		debug_print_dims(DP_DEBUG4, DIMS, pos);

		if (dims[READ_DIM] != mdh.samples) {

			debug_printf(DP_WARN, "Wrong number of samples: %d != %d.\n", dims[READ_DIM], mdh.samples);
			return -1;
		}

		if ((0 != mdh.channels) && (dims[COIL_DIM] != mdh.channels)) {

			debug_printf(DP_WARN, "Wrong number of channels: %d != %d.\n", dims[COIL_DIM], mdh.channels);
			return -1;
		}

		xread(fd, buf + pos[COIL_DIM] * dims[READ_DIM], dims[READ_DIM] * CFL_SIZE);
	}

	pos[COIL_DIM] = 0;
	return 0;
}


// TODO: in the case of "cfl" file output,
// don't add ".cfl" to the name.
static char* create_bc_name(const char* name)
{
	char* last_name = strrchr(name, '.');

	bool true_last_name = (NULL!=last_name) && (name!=last_name) && ((0==strcmp(last_name, ".ra")) || (0==strcmp(last_name, ".coo")) || (0==strcmp(last_name, ".mem")));

	if (!true_last_name)
		last_name = "";

	size_t vor_len = strlen(name) - strlen(last_name);

	const char* addi_name = "_BC";

	char* bc_name = (char *)malloc(vor_len + strlen(addi_name) + strlen(last_name));

	memcpy(bc_name, name, vor_len);
	bc_name[vor_len] = '\0';
	strcat(bc_name, addi_name);
	strcat(bc_name, last_name);

	return bc_name;
}




static const char usage_str[] = "<dat file> <output> [<pmu>]";
//	fprintf(fd, "Usage: %s [...] [-a A] <dat file> <output>\n", name);

static const char help_str[] = "Read data from Siemens twix (.dat) files.";


int main_twixread(int argc, char* argv[argc])
{
	long adcs = 0;
	long radial_lines = -1;
	long bc_scans = 0;
	long bc_adcs = 0;

	bool autoc = false;
	bool linectr = false;
	bool partctr = false;
	bool mpi = false;
	bool out_pmu = false;
	bool out_timestamp = false;

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);

	struct opt_s opts[] = {

		OPT_LONG('x', &(dims[READ_DIM]), "X", "number of samples (read-out)"),
		OPT_LONG('r', &radial_lines, "R", "radial lines"),
		OPT_LONG('y', &(dims[PHS1_DIM]), "Y", "phase encoding steps"),
		OPT_LONG('z', &(dims[PHS2_DIM]), "Z", "partition encoding steps"),
		OPT_LONG('s', &(dims[SLICE_DIM]), "S", "number of slices"),
		OPT_LONG('v', &(dims[AVG_DIM]), "V", "number of averages"),
		OPT_LONG('c', &(dims[COIL_DIM]), "C", "number of channels"),
		OPT_LONG('n', &(dims[TIME_DIM]), "N", "number of repetitions"),
		OPT_LONG('p', &(dims[COEFF_DIM]), "P", "number of cardiac phases"),
		OPT_LONG('f', &(dims[TIME2_DIM]), "F", "number of flow encodings"),
		OPT_LONG('b', &bc_scans, "B", "number of body-coil scans"),
		OPT_LONG('i', &(dims[LEVEL_DIM]), "I", "number inversion experiments"),
		OPT_LONG('a', &adcs, "A", "total number of ADCs"),
		OPT_SET('A', &autoc, "automatic [guess dimensions]"),
		OPT_SET('L', &linectr, "use linectr offset"),
		OPT_SET('P', &partctr, "use partctr offset"),
		OPT_SET('M', &mpi, "MPI mode"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (-1 != radial_lines)
		dims[PHS1_DIM] = radial_lines;

	if (0 == adcs)
		adcs = dims[PHS1_DIM] * dims[PHS2_DIM] * dims[SLICE_DIM] * dims[TIME_DIM] *  dims[TIME2_DIM] * dims[LEVEL_DIM];



	long bc_dims[DIMS] = {[0 ... DIMS - 1] = 0};

	md_copy_dims(DIMS, bc_dims, dims);
	bc_dims[COIL_DIM] = 2;
	bc_dims[TIME_DIM] = bc_scans;

	bc_adcs = bc_dims[PHS1_DIM] * bc_dims[PHS2_DIM] * bc_dims[SLICE_DIM] * bc_dims[TIME_DIM] * bc_dims[TIME2_DIM] * dims[LEVEL_DIM];

	if (0 < bc_scans) {

		radial_lines = -1;
		autoc = false;
		mpi = false;
	}



	debug_printf(DP_DEBUG1, "bodycoil dims:\t");
	debug_print_dims(DP_DEBUG1, DIMS, bc_dims);

	debug_printf(DP_DEBUG1, "measured dims:\t");
	debug_print_dims(DP_DEBUG1, DIMS, dims);



	int ifd;
	if (-1 == (ifd = open(argv[1], O_RDONLY)))
		error("error opening file.");

	struct hdr_s hdr;
	bool vd = siemens_meas_setup(ifd, &hdr);

	long off[DIMS] = { 0 };

	if (autoc) {

		long max[DIMS] = { [COIL_DIM] = 1000 };
		long min[DIMS] = { 0 }; // min is always 0

		adcs = 0;

		while (true) {

			if (-1 == siemens_bounds(vd, ifd, min, max))
				break;

			debug_print_dims(DP_DEBUG3, DIMS, max);

			adcs++;
		}

		for (unsigned int i = 0; i < DIMS; i++) {

			off[i] = -min[i];
			dims[i] = max[i] + off[i];
		}

		debug_printf(DP_DEBUG2, "Dimensions: ");
		debug_print_dims(DP_DEBUG2, DIMS, dims);
		debug_printf(DP_DEBUG2, "Offset: ");
		debug_print_dims(DP_DEBUG2, DIMS, off);

		siemens_meas_setup(ifd, &hdr); // reset
	}

	

	long odims[DIMS];
	md_copy_dims(DIMS, odims, dims);

	if (-1 != radial_lines) {

		// change output dims (must have identical layout!)
		odims[0] = 1;
		odims[1] = dims[0];
		odims[2] = dims[1];
		assert(1 == dims[2]);
	}



	complex float* out = create_cfl(argv[2], DIMS, odims);
	md_clear(DIMS, odims, out, CFL_SIZE);



	long bc_odims[DIMS];
	md_copy_dims(DIMS, bc_odims, bc_dims);
	
	complex float* bc_out = NULL;

	if (0 < bc_scans)
		bc_out = create_cfl(create_bc_name(argv[2]), DIMS, bc_odims);


	long pmu_dims[DIMS];
	md_select_dims(DIMS, ~(READ_FLAG|COIL_FLAG), pmu_dims, dims);

	complex float* pmu;
	complex float* timestamp;

	if (argc > 3) {

		out_pmu = true;

		pmu = create_cfl(argv[3], DIMS, pmu_dims);
		md_clear(DIMS, pmu_dims, pmu, CFL_SIZE);
	}

	if (argc > 4) {

		out_timestamp = true;

		timestamp = create_cfl(argv[4], DIMS, pmu_dims);
		md_clear(DIMS, pmu_dims, timestamp, CFL_SIZE);
	}



	uint32_t buf_pmu;
	complex float* val_pmu;
	long val_pmu_dims[DIMS] = { [0 ... DIMS - 1] = 1};

	if (out_pmu)
		val_pmu = md_alloc(DIMS, val_pmu_dims, CFL_SIZE);

	uint32_t buf_timestamp;
	complex float* val_timestamp;
	long val_timestamp_dims[DIMS] = { [0 ... DIMS - 1] = 1};

	if (out_timestamp)
		val_timestamp = md_alloc(DIMS, val_timestamp_dims, CFL_SIZE);



	debug_printf(DP_DEBUG1, "___ reading BC data.\n");

	long bc_adc_dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|COIL_FLAG, bc_adc_dims, bc_dims);

	void* bc_buf = md_alloc(DIMS, bc_adc_dims, CFL_SIZE);

	while (bc_adcs--) {

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		if (-1 == siemens_adc_read(vd, ifd, linectr, partctr, bc_dims, pos, bc_buf, &buf_pmu, &buf_timestamp)) {

			debug_printf(DP_WARN, "Stopping.\n");
			break;
		}

		for (unsigned int i = 0; i < DIMS; i++)
			pos[i] += off[i];

		debug_print_dims(DP_DEBUG1, DIMS, pos);

		if (!md_is_index(DIMS, pos, bc_dims)) {

			debug_printf(DP_WARN, "Index out of bounds.\n");
			debug_printf(DP_WARN, " bc_dims: ");
			debug_print_dims(DP_WARN, DIMS, bc_dims);
			debug_printf(DP_WARN, "     pos: ");
			debug_print_dims(DP_WARN, DIMS, pos);
			continue;
		}

		md_copy_block(DIMS, pos, bc_dims, bc_out, bc_adc_dims, bc_buf, CFL_SIZE); 
	}

	md_free(bc_buf);
	
	if (0 < bc_scans)
		unmap_cfl(DIMS, bc_odims, bc_out);



	debug_printf(DP_DEBUG1, "___ reading measured data.\n");


	long adc_dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|COIL_FLAG, adc_dims, dims);

	void* buf = md_alloc(DIMS, adc_dims, CFL_SIZE);


	long mpi_slice = -1;
	long multi_inv = -1;

	while (adcs--) {

		long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

		if (-1 == siemens_adc_read(vd, ifd, linectr, partctr, dims, pos, buf, &buf_pmu, &buf_timestamp)) {

			debug_printf(DP_WARN, "Stopping.\n");
			break;
		}

		for (unsigned int i = 0; i < DIMS; i++)
			pos[i] += off[i];

		if (mpi) {

			pos[SLICE_DIM] = mpi_slice;

			if ((0 == pos[TIME_DIM]) && (0 == pos[PHS1_DIM]))
				mpi_slice++;
		}

		if (1 < dims[LEVEL_DIM]) {

			pos[LEVEL_DIM] = multi_inv;

			if ((0 == pos[TIME_DIM]) && (0 == pos[PHS1_DIM]))
				multi_inv++;
		}

		debug_print_dims(DP_DEBUG1, DIMS, pos);

		if (!md_is_index(DIMS, pos, dims)) {

			debug_printf(DP_WARN, "Index out of bounds.\n");
			debug_printf(DP_WARN, " dims: ");
			debug_print_dims(DP_WARN, DIMS, dims);
			debug_printf(DP_WARN, "  pos: ");
			debug_print_dims(DP_WARN, DIMS, pos);
			continue;
		}

		md_copy_block(DIMS, pos, dims, out, adc_dims, buf, CFL_SIZE); 

		if (out_pmu) {

			*val_pmu = (complex float) buf_pmu + 0 * 1i;

			//debug_printf(DP_INFO, "val_pmu: %f\n", creal(*val_pmu));
			md_copy_block(DIMS, pos, pmu_dims, pmu, val_pmu_dims, val_pmu, CFL_SIZE);
		}

		if (out_timestamp) {

			*val_timestamp = (complex float) buf_timestamp + 0 * 1i;

			//debug_printf(DP_INFO, "val_pmu: %f\n", creal(*val_pmu));
			md_copy_block(DIMS, pos, pmu_dims, timestamp, val_timestamp_dims, val_timestamp, CFL_SIZE);
		}

	}

	md_free(buf);
	unmap_cfl(DIMS, odims, out);

	if (out_pmu) {

		unmap_cfl(DIMS, pmu_dims, pmu);
		md_free(val_pmu);
	}

	if (out_timestamp) {

		unmap_cfl(DIMS, pmu_dims, timestamp);
		md_free(val_timestamp);
	}

	return 0;
}

