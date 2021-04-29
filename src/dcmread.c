
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>

#include "misc/dicom.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

static const char help_str[] = "";

int main_dcmread(int argc, char* argv[argc])
{
	const char* dcm_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &dcm_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	int dims[2];
	unsigned char* img = dicom_read(dcm_file, dims);

	if (NULL == img)
		error("reading dicom file '%s'", dcm_file);

	printf("Size: %d-%d\n", dims[0], dims[1]);

	long d[2] = { dims[0], dims[1] };
	complex float* out = create_cfl(out_file, 2, d);
	
	for (int j = 0; j < dims[1]; j++)
		for (int i = 0; i < dims[0]; i++)
			out[j * dims[0] + i] = (img[(i * dims[1] + j) * 2 + 0]
						+ (img[(i * dims[1] + j) * 2 + 1] << 8))
						/ 65535.;

	xfree(img);
	unmap_cfl(2, d, out);
	exit(0);
}
