
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <fcntl.h>
#include <unistd.h>

#include "misc/dicom.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char* help_str = "";

int main_dcmtag(int argc, char* argv[])
{
	const char* xy_str = NULL;
	const char* dcm_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_STRING(true, &xy_str, "0000,0000"),
		ARG_STRING(true, &dcm_file, "<dmc>"),
		ARG_STRING(false, &out_file, "<out>"),
	};

	const char* repr = NULL;

	const struct opt_s opts[] = {

		OPT_STRING('r', &repr, "XX", "value representation"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	
	int x, y, n;
	int ret = sscanf(xy_str, "%x,%x%n", &x, &y, &n);

	if (   (ret != 2) || (n != (int)strlen(xy_str))
	    || (0 > x) || (x > 0xFFFF)
	    || (0 > y) || (y > 0xFFFF))
		error("invalid dicom tag\n");


	struct dicom_obj_s* dobj = dicom_open(dcm_file);

	if (NULL == dobj)
		error("reading dicom file '%s'\n", dcm_file);


	struct element el = { .tag = { (uint16_t)x, (uint16_t)y }, .vr = "--" };

	if (NULL != repr) {

		if (2 != strlen(repr))
			error("incorrect value representation\n");

		el.vr[0] = repr[0];
		el.vr[1] = repr[1];
	}

	dicom_query_tags(dobj, 1, &el);

	if (NULL == el.data)
		error("tag not found\n");

	if (NULL != out_file) {

		int fd = open(out_file, O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR);

		if (-1 == fd)
			error("error opening file '%s'\n", out_file);

		if (el.len != write(fd, el.data, el.len))
			error("error writing\n");

		close(fd);
		goto end;
	}


#define VR2(a, b) (a * 256 + b)
#define VR(x) VR2((x)[0], (x)[1])

	switch (VR(el.vr)) {
		const char* str;

	case VR2('D', 'A'):	// date
	case VR2('D', 'T'):	// date time
	case VR2('T', 'M'):	// time

	case VR2('P', 'N'):	// person name
	case VR2('A', 'S'):	// age string
	case VR2('D', 'S'):	// decimal string
	case VR2('I', 'S'):	// integer string
	case VR2('C', 'S'):	// code string
	case VR2('L', 'T'):	// long text
	case VR2('S', 'T'):	// short text
	case VR2('U', 'T'):	// unlimited text

	case VR2('U', 'I'):	// unique identifier

		str = strndup(el.data, el.len);
		bart_printf("%s\n", str);
		xfree(str);
		break;

	case VR2('F', 'L'):	// IEEE 754:1985 32 bit
		bart_printf("%f\n", (double)(*(float*)el.data));
		break;

	case VR2('F', 'D'):	// IEEE 754:1985 64 bit
		bart_printf("%f\n", *(double*)el.data);
		break;

	case VR2('U', 'S'):	// unsigned short
		bart_printf("%d\n", *(uint16_t*)el.data);
		break;

	case VR2('S', 'L'):	// signed long
		bart_printf("%d\n", *(uint32_t*)el.data);
		break;

	default:
		error("unsupported element type: %2s\n", el.vr);
	}

end:
	dicom_close(dobj);

	exit(0);
}
