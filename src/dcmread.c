
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>

#include "misc/dicom.h"
#include "misc/mmio.h"
#include "misc/misc.h"



int main_dcmread(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, 2, "", "");

	int dims[2];
	unsigned char* img = dicom_read(argv[1], dims);

	if (NULL == img)
		error("reading dicom file '%s'", argv[1]);

	printf("Size: %d-%d\n", dims[0], dims[1]);

	long d[2] = { dims[0], dims[1] };
	complex float* out = create_cfl(argv[2], 2, d);
	
	for (int j = 0; j < dims[1]; j++)
		for (int i = 0; i < dims[0]; i++)
			out[j * dims[0] + i] = (img[(i * dims[1] + j) * 2 + 0]
						+ (img[(i * dims[1] + j) * 2 + 1] << 8))
						/ 65535.;

	xfree(img);
	unmap_cfl(2, d, out);
	exit(0);
}
