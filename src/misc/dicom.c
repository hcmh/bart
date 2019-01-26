/* Copyright 2015. The Regents of the University of California.
 * Copyright 2018-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2015-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

/* NOTE: This code packs pixel data into very simple dicom images
 * with only image related tags. Other mandatory DICOM tags are
 * missing. We only support 16 bit little endian gray scale images.
 *
 * FOR RESEARCH USE ONLY - NOT FOR DIAGNOSTIC USE
 */

#define _GNU_SOURCE

#include <limits.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <locale.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef HAVE_UUID
#include <uuid/uuid.h>
#endif

#include "misc/misc.h" // for error

#include "dicom.h"



// US unsigned short
// LS unsigned long
// IS integer string
// LT long text
// CS code string
// OW other word string

#define DGRP_IMAGE		0x0028
#define DTAG_IMAGE_SAMPLES_PER_PIXEL	0x0002
#define DTAG_IMAGE_PHOTOM_INTER		0x0004
#define DTAG_IMAGE_ROWS			0x0010
#define DTAG_IMAGE_COLS			0x0011
#define DTAG_IMAGE_BITS_ALLOC		0x0100
#define DTAG_IMAGE_BITS_STORED		0x0101
#define DTAG_IMAGE_PIXEL_HIGH_BIT	0x0102
#define DTAG_IMAGE_PIXEL_REP		0x0103	// 0 unsigned 2 two's complement

#define MONOCHROME2			"MONOCHROME2"

#define DGRP_PIXEL			0x7FE0
#define DTAG_PIXEL_DATA			0x0010

#define DGRP_FILE		0x0002
#define DTAG_META_SIZE			0x0000
#define DTAG_TRANSFER_SYNTAX		0x0010
#define LITTLE_ENDIAN_EXPLICIT		"1.2.840.10008.1.2.1"
#define LITTLE_ENDIAN_IMPLICIT		"1.2.840.10008.1.2"

#define DGRP_IMAGE2		0x0020
#define DTAG_IMAGE_INSTANCE_NUM		0x0013
#define DTAG_COMMENT			0x4000

#define DTAG_STUDY_INSTANCE_UID		0x000D
#define DTAG_SERIES_INSTANCE_UID	0x000E

#define DTAG_POSITION			0x0032
#define DTAG_ORIENTATION		0x0037

#define DGRP_SEQ			0xFFFE
#define DTAG_SEQ_ITEM			0xE000
#define DTAG_SEQ_ITEM_DELIM		0xE00D
#define DTAG_SEQ_DELIM			0xE0DD

// order matters...
enum eoffset {

	ITAG_META_SIZE,
	ITAG_TRANSFER_SYNTAX, 
#ifdef HAVE_UUID
	ITAG_STUDY_INSTANCE_UID,
	ITAG_SERIES_INSTANCE_UID,
#endif
	ITAG_IMAGE_INSTANCE_NUM, 
	ITAG_COMMENT,
	ITAG_IMAGE_SAMPLES_PER_PIXEL, 
	ITAG_IMAGE_PHOTOM_INTER, 
	ITAG_IMAGE_ROWS, 
	ITAG_IMAGE_COLS, 
	ITAG_IMAGE_BITS_ALLOC, 
	ITAG_IMAGE_BITS_STORED, 
	ITAG_IMAGE_PIXEL_HIGH_BIT, 
	ITAG_IMAGE_PIXEL_REP, 
	ITAG_PIXEL_DATA, 

	NR_ENTRIES,
};


struct tag {

	uint16_t group;
	uint16_t element;
};

struct element {

	struct tag tag;
	char vr[2];

	unsigned int len;
	const void* data;
};



struct element dicom_elements_default[] = {

	[ITAG_META_SIZE] = { { DGRP_FILE, DTAG_META_SIZE }, "UL", 4, &(uint32_t){ 28 } },
	[ITAG_TRANSFER_SYNTAX] = { { DGRP_FILE, DTAG_TRANSFER_SYNTAX }, "UI", sizeof(LITTLE_ENDIAN_EXPLICIT), LITTLE_ENDIAN_EXPLICIT },
#ifdef HAVE_UUID
	[ITAG_STUDY_INSTANCE_UID] = { { DGRP_IMAGE2, DTAG_STUDY_INSTANCE_UID }, "UI", 0, NULL },
	[ITAG_SERIES_INSTANCE_UID] = { { DGRP_IMAGE2, DTAG_SERIES_INSTANCE_UID }, "UI", 0, NULL },
#endif
	[ITAG_IMAGE_INSTANCE_NUM] = { { DGRP_IMAGE2, DTAG_IMAGE_INSTANCE_NUM }, "IS", 0, NULL },
	[ITAG_COMMENT] = { { DGRP_IMAGE2, DTAG_COMMENT }, "LT", 22, "NOT FOR DIAGNOSTIC USE\0\0" },
	[ITAG_IMAGE_SAMPLES_PER_PIXEL] = { { DGRP_IMAGE, DTAG_IMAGE_SAMPLES_PER_PIXEL }, "US", 2, &(uint16_t){ 1 } }, 		// gray scale
	[ITAG_IMAGE_PHOTOM_INTER] = { { DGRP_IMAGE, DTAG_IMAGE_PHOTOM_INTER }, "CS", sizeof(MONOCHROME2), MONOCHROME2 },	// 0 is black
	[ITAG_IMAGE_ROWS] = { { DGRP_IMAGE, DTAG_IMAGE_ROWS }, "US", 2, NULL },
	[ITAG_IMAGE_COLS] = { { DGRP_IMAGE, DTAG_IMAGE_COLS }, "US", 2, NULL },
	[ITAG_IMAGE_BITS_ALLOC] = { { DGRP_IMAGE, DTAG_IMAGE_BITS_ALLOC }, "US", 2, &(uint16_t){ 16 } },			//
	[ITAG_IMAGE_BITS_STORED] = { { DGRP_IMAGE, DTAG_IMAGE_BITS_STORED }, "US", 2, &(uint16_t){ 16 } },			// 12 for CT
	[ITAG_IMAGE_PIXEL_HIGH_BIT] = { { DGRP_IMAGE, DTAG_IMAGE_PIXEL_HIGH_BIT }, "US", 2, &(uint16_t){ 15 } },
	[ITAG_IMAGE_PIXEL_REP] = { { DGRP_IMAGE, DTAG_IMAGE_PIXEL_REP }, "US", 2, &(uint16_t){ 0 } },			// unsigned
	[ITAG_PIXEL_DATA] = { { DGRP_PIXEL, DTAG_PIXEL_DATA }, "OW", 0, NULL },
};




static bool vr_oneof(const char a[2], unsigned int N, const char b[N][2])
{
	for (unsigned int i = 0; i < N; i++)
		if ((a[0] == b[i][0]) && (a[1] == b[i][1]))
			return true;

	return false;
}

static int dicom_write_element(unsigned int len, char buf[static 8 + len], struct element e)
{
	assert((((union { uint16_t s; uint8_t b; }){ 1 }).b));	// little endian

	assert(len == e.len);
	assert(0 == len % 2);

	int o = 0;

	buf[o++] = ((e.tag.group >> 0) & 0xFF);
	buf[o++] = ((e.tag.group >> 8) & 0xFF);

	buf[o++] = ((e.tag.element >> 0) & 0xFF);
	buf[o++] = ((e.tag.element >> 8) & 0xFF);

 	buf[o++] = e.vr[0];
	buf[o++] = e.vr[1];

	if (!vr_oneof(e.vr, 5, (const char[5][2]){ "OB", "OW", "SQ", "UN", "UT" })) {

		buf[o++] = ((len >> 0) & 0xFF);
		buf[o++] = ((len >> 8) & 0xFF);
	
	} else {
	
		buf[o++] = 0; // reserved
		buf[o++] = 0; // reserved
		buf[o++] = ((len >>  0) & 0xFF);
		buf[o++] = ((len >>  8) & 0xFF);
		buf[o++] = ((len >> 16) & 0xFF);
		buf[o++] = ((len >> 24) & 0xFF);
	}

	memcpy(buf + o, e.data, len);
	return len + o;
}

static int dicom_read_sequence(unsigned int len, const unsigned char buf[len], bool use_implicit);

static int dicom_read_element(struct element* e, unsigned int len, const unsigned char buf[len])
{
	assert((((union { uint16_t s; uint8_t b; }){ 1 }).b));	// little endian

	int o = 0;

	e->tag.group  = ((uint16_t)(buf[o++])) << 0;
	e->tag.group |= ((uint16_t)(buf[o++])) << 8;

	e->tag.element  = ((uint16_t)(buf[o++])) << 0;
	e->tag.element |= ((uint16_t)(buf[o++])) << 8;

	e->vr[0] = buf[o++];
	e->vr[1] = buf[o++];

	if (!vr_oneof(e->vr, 5, (const char[5][2]){ "OB", "OW", "SQ", "UN", "UT" })) {

		e->len  = ((uint16_t)(buf[o++])) << 0;
		e->len |= ((uint16_t)(buf[o++])) << 8;

	} else {

		(void)buf[o++]; // reserved
		(void)buf[o++]; // reserved

		e->len  = ((uint16_t)(buf[o++])) <<  0;
		e->len |= ((uint16_t)(buf[o++])) <<  8;
		e->len |= ((uint16_t)(buf[o++])) << 16;
		e->len |= ((uint16_t)(buf[o++])) << 24;
	}

	e->data = buf + o;

	if (   vr_oneof(e->vr, 1, (const char[1][2]){ "SQ" })
	    && (e->len == 0xFFFFFFFF))
		e->len = dicom_read_sequence(len - o, e->data, false);

	return o + e->len;
}

static int dicom_read_implicit(struct element* e, unsigned int len, const unsigned char buf[len])
{
	assert((((const union { uint16_t s; uint8_t b; }){ 1 }).b));	// little endian

	int o = 0;

	e->tag.group  = ((uint16_t)(buf[o++])) << 0;
	e->tag.group |= ((uint16_t)(buf[o++])) << 8;

	e->tag.element  = ((uint16_t)(buf[o++])) << 0;
	e->tag.element |= ((uint16_t)(buf[o++])) << 8;

	e->len  = ((uint16_t)(buf[o++])) <<  0;
	e->len |= ((uint16_t)(buf[o++])) <<  8;
	e->len |= ((uint16_t)(buf[o++])) << 16;
	e->len |= ((uint16_t)(buf[o++])) << 24;

	e->data = buf + o;

	if (e->len == 0xFFFFFFFF)	// let's just assume VR SQ
		e->len = dicom_read_sequence(len - o, e->data, true);

	return o + e->len;
}

static int dicom_read_seq(struct element* e, unsigned int len, const unsigned char buf[len])
{
	assert((((const union { uint16_t s; uint8_t b; }){ 1 }).b));	// little endian

	int o = 0;

	e->tag.group  = ((uint16_t)(buf[o++])) << 0;
	e->tag.group |= ((uint16_t)(buf[o++])) << 8;

	e->tag.element  = ((uint16_t)(buf[o++])) << 0;
	e->tag.element |= ((uint16_t)(buf[o++])) << 8;

	e->len  = ((uint16_t)(buf[o++])) <<  0;
	e->len |= ((uint16_t)(buf[o++])) <<  8;
	e->len |= ((uint16_t)(buf[o++])) << 16;
	e->len |= ((uint16_t)(buf[o++])) << 24;

	e->data = buf + o;

	return o;
}



static int dicom_tag_compare(const struct tag a, const struct tag b)
{
	if (a.group == b.group)
		return a.element - b.element;

	return a.group - b.group;
}



static int dicom_query(size_t len, const unsigned char buf[len], bool use_implicit, int N, struct element ellist[N]);
static int dicom_read_sequence(unsigned int len, const unsigned char buf[len], bool use_implicit)
{
	struct element e;

	int o = 0;

	do {
		o += dicom_read_seq(&e, len - o, buf + o);

		if (0 == dicom_tag_compare(e.tag, (struct tag){ DGRP_SEQ, DTAG_SEQ_DELIM })) {

			assert(0 == e.len);
			break;
		}

		assert(0 == dicom_tag_compare(e.tag, (struct tag){ DGRP_SEQ, DTAG_SEQ_ITEM }));

		if (0xFFFFFFFF == e.len) {

			struct element end[1] = {
				{ { DGRP_SEQ, DTAG_SEQ_ITEM_DELIM }, "--", 0, NULL },
			};

			assert(use_implicit);	// FIXME: explicit
			o += dicom_query(len - o, e.data, use_implicit, 1, end);

			assert(0 == end[0].len);

		} else {

			o += e.len;
		}

	} while (true);

	return o;
}




static int double_dabble(int l, char bcd[l], int n, const unsigned char in[n])
{
	int s = l - 1;

	_Static_assert(CHAR_BIT == 8, "bits per char not 8");

	for (int i = 0; i < n; i++) {

		for (int j = 0; j < 8; j++) {	// 8 bits per char

			for (int k = s; k < l; k++)
				if (bcd[k] % 16 >= 5)
					bcd[k] += 3;

			if (bcd[s] >= 8)
				s--;

			for (int k = s; k < l; k++) {

				bcd[k] *= 2;
				bcd[k] %= 16;
				bcd[k] |= (k == l - 1)
						? (0 != (in[i] & (1 << (7 - j))))	// 8 bits per char
						: (bcd[k + 1] >= 8);
			}
		}
	}

	return s;
}

void dicom_generate_uid(char buf[64])
{
	assert(NULL != buf);

	memset(buf, 0, 64);
	strcpy(buf, "2.25.");


#ifndef HAVE_UUID
	(void)double_dabble;
	assert(0);
#else
	uuid_t uuid;
	uuid_generate(uuid);

#if 0
	char hex[40];
	uuid_unparse(uuid, hex);
	printf("%s\n", hex);
	// convert digits back to hex: bc <<< "obase=16; XXXXX"
#endif

	char digits[41] = { 0 };
	int s = double_dabble(40, digits, 16, uuid);

	for (int j = 0; j < 40; j++)
		digits[j] += '0';

	while ('0' == digits[s])
		s++;

	assert('\0' != digits[s]);

	strcpy(buf + 5, digits + s);
#endif
}

static int dicom_len(const char* x)
{
	int len = strlen(x);

	assert(0 == x[len]);

	if (1 == len % 2)
		len++;

	assert(0 == x[len]);

	return len;
}


int dicom_write(const char* name, const char study_uid[64], const char series_uid[64], unsigned int cols, unsigned int rows, long inum, const unsigned char* img)
{
	int fd;
	void* addr;
	struct stat st;
	int ret = -1;

	if (-1 == (fd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)))
		goto cleanup;

	if (-1 == fstat(fd, &st))
		goto cleanup;

	assert(NR_ENTRIES * sizeof(struct element) == sizeof(dicom_elements_default));

	struct element dicom_elements[NR_ENTRIES];

	memcpy(&dicom_elements, &dicom_elements_default, sizeof(dicom_elements_default));

	dicom_elements[ITAG_IMAGE_ROWS].data = &(uint16_t){ rows };
	dicom_elements[ITAG_IMAGE_COLS].data = &(uint16_t){ cols };
#ifdef HAVE_UUID
	dicom_elements[ITAG_STUDY_INSTANCE_UID].data = study_uid;
	dicom_elements[ITAG_STUDY_INSTANCE_UID].len = dicom_len(study_uid);

	dicom_elements[ITAG_SERIES_INSTANCE_UID].data = series_uid;
	dicom_elements[ITAG_SERIES_INSTANCE_UID].len = dicom_len(series_uid);
#else
	(void)study_uid;
	(void)series_uid;
#endif
	char inst_num[12] = { 0 }; // max number of bytes for InstanceNumber tag
	sprintf(inst_num, "+%04ld", inum);

	dicom_elements[ITAG_IMAGE_INSTANCE_NUM].data = inst_num;
	dicom_elements[ITAG_IMAGE_INSTANCE_NUM].len = dicom_len(inst_num);


	dicom_elements[ITAG_PIXEL_DATA].data = img;
	dicom_elements[ITAG_PIXEL_DATA].len = 2 * rows * cols;

	size_t size = 128 + 4;

	size += 4;	// the pixel data element is larger

	for (int i = 0; i < NR_ENTRIES; i++)
		size += 8 + dicom_elements[i].len;

	if (-1 == ftruncate(fd, size))
		goto cleanup;

	if (MAP_FAILED == (addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)))
		goto cleanup;


	// write header

	memset(addr, 0, 128);
	memcpy(addr + 128, "DICM", 4);

	size_t off = 128 + 4;

	struct tag last = { 0, 0 };

	// make sure tags are in ascending order
	for (int i = 0; i < NR_ENTRIES; i++) {

		assert(0 > dicom_tag_compare(last, dicom_elements[i].tag));

		last = dicom_elements[i].tag;

		off += dicom_write_element(dicom_elements[i].len, addr + off, dicom_elements[i]);
	}

	assert(0 == size - off);

	ret = 0;

	if (-1 == munmap((void*)addr, size))
		error("abort!");

cleanup:
	if (-1 == close(fd))
		error("abort!");

	return ret;
}


static int dicom_query(size_t len, const unsigned char buf[len], bool use_implicit, int N, struct element ellist[N])
{
	size_t off = 0;

	for (int i = 0; i < N; i++) {

		struct element element;

		do {
			size_t l;

			assert(off < len);

			l = (use_implicit ? dicom_read_implicit : dicom_read_element)(&element, len - off, buf + off);

			off += l;

		} while(0 > dicom_tag_compare(element.tag,
				ellist[i].tag));

		if (0 == dicom_tag_compare(element.tag, ellist[i].tag))
			memcpy(&ellist[i], &element, sizeof(element));
	}

	return off;
}



struct dicom_obj_s {

	void* data;
	size_t size;
	off_t off;
	bool implicit;
};

struct dicom_obj_s* dicom_open(const char* name)
{
	int fd;
	void* addr;
	struct stat st;

	if (-1 == (fd = open(name, O_RDONLY)))
		goto cleanup;

	if (-1 == fstat(fd, &st))
		goto cleanup;

	size_t size = st.st_size;

	if (MAP_FAILED == (addr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0)))
		goto cleanup;

	size_t off = 128;

	unsigned char* buf = addr;

	if (0 != memcmp("DICM", buf + off, 4))
		goto cleanup2;

	off += 4;

	size_t len = st.st_size;

	struct element dicom_elements[NR_ENTRIES];
	memcpy(dicom_elements, dicom_elements_default, sizeof(dicom_elements_default));

	bool implicit = false;

	// read META TAGS
	if (0 == dicom_query(len - off, buf + off, false, 2, dicom_elements))	// FIXME: condition
		goto cleanup2;


	off += 12; // size of meta tag
	off += *(uint32_t*)dicom_elements[ITAG_META_SIZE].data;
	implicit = (0 == memcmp(dicom_elements[ITAG_TRANSFER_SYNTAX].data, LITTLE_ENDIAN_IMPLICIT, dicom_elements[ITAG_TRANSFER_SYNTAX].len)); // FIXME

	struct dicom_obj_s* dobj = xmalloc(sizeof(struct dicom_obj_s));

	dobj->data = buf;
	dobj->size = size;
	dobj->off = off;
	dobj->implicit = implicit;

	return dobj;

cleanup2:
	if (-1 == munmap((void*)addr, size))
		abort();

cleanup:
	if (-1 == close(fd))
		abort();

	return NULL;
}

void dicom_close(const struct dicom_obj_s* dobj)
{
	if (-1 == munmap(dobj->data, dobj->size))
		abort();

	xfree(dobj);
}


static int dicom_query_tags(const struct dicom_obj_s* dobj, int N, struct element ellist[N])
{
	off_t off = dobj->off;

	return dicom_query(dobj->size - off, dobj->data + off, dobj->implicit, N, ellist);
}


int dicom_instance_num(const struct dicom_obj_s* dobj)
{
	struct element instance_num;
	memcpy(&instance_num, &dicom_elements_default[ITAG_IMAGE_INSTANCE_NUM], sizeof(struct element));

	if (0 == dicom_query_tags(dobj, 1, &instance_num))
		return 0;

	assert(NULL != instance_num.data);
	assert(instance_num.len < 32);

	char copy[32] = { 0 };
	strncpy(copy, instance_num.data, instance_num.len);

	return atoi(copy);
}


void dicom_geometry(const struct dicom_obj_s* dobj, float pos[3][3])
{
	struct element dicom_elements[] = {

		{ { DGRP_IMAGE2, DTAG_POSITION }, "DS", 0, NULL },
		{ { DGRP_IMAGE2, DTAG_ORIENTATION }, "DS", 0, NULL },
	};

	if (0 == dicom_query_tags(dobj, ARRAY_SIZE(dicom_elements), dicom_elements))
		return;

	assert(NULL != dicom_elements[0].data);
	assert(NULL != dicom_elements[1].data);

	locale_t nlc = newlocale(LC_NUMERIC, "C", NULL);
	locale_t oldlc = uselocale(nlc);


	assert(dicom_elements[0].len < 51);
	char tmp1[51] = { 0 };
	strncpy(tmp1, dicom_elements[0].data, dicom_elements[0].len);

	sscanf(tmp1, "%f\\%f\\%f", &pos[0][0], &pos[0][1], &pos[0][2]);

	assert(dicom_elements[1].len < 102);
	char tmp2[102] = { 0 };

	strncpy(tmp2, dicom_elements[1].data, dicom_elements[1].len);

	sscanf(tmp2, "%f\\%f\\%f\\%f\\%f\\%f",
		&pos[1][0], &pos[1][1], &pos[1][2],
		&pos[2][0], &pos[2][1], &pos[2][2]);

	uselocale(oldlc);
	freelocale(nlc);
}


unsigned char* dicom_read_image(const struct dicom_obj_s* dobj, int dims[2])
{
	unsigned char* ret = NULL;
	int skip = 6;

	struct element dicom_elements[NR_ENTRIES];
	memcpy(dicom_elements, dicom_elements_default, sizeof(dicom_elements_default));


	if (0 == dicom_query_tags(dobj, NR_ENTRIES - skip, dicom_elements + skip))
		goto cleanup;

	if (   (NULL == dicom_elements[ITAG_IMAGE_ROWS].data)
	    || (NULL == dicom_elements[ITAG_IMAGE_COLS].data)
	    || (NULL == dicom_elements[ITAG_PIXEL_DATA].data))
		goto cleanup;

	int rows = *(uint16_t*)dicom_elements[ITAG_IMAGE_ROWS].data;
	int cols = *(uint16_t*)dicom_elements[ITAG_IMAGE_COLS].data;

	dims[0] = rows;
	dims[1] = cols;

	ret = xmalloc(2 * rows * cols);

	if (ret)
		memcpy(ret, dicom_elements[ITAG_PIXEL_DATA].data, 2 * rows * cols);
cleanup:
	return ret;
}

unsigned char* dicom_read(const char* name, int dims[2])
{
	struct dicom_obj_s* dobj;

	if (NULL == (dobj = dicom_open(name)))
		return NULL;

	unsigned char* ret = dicom_read_image(dobj, dims);

	dicom_close(dobj);

	return ret;
}


