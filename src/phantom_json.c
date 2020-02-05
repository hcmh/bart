
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/cJSON.h"
#include "simu/read_json.h"

#include "simu/phantom.h"
#include "simu/simulation.h"
#include "simu/shepplogan.h"
#include "simu/bloch.h"



/*
 * Example of *.json file to pass geomtry to phantom_json.c
{
	"objects": [
		{
			"type": "ellipse",
			"intensity": 1,
			"axis": [0.75, 0.75],
			"center": [0, 0],
			"angle": 0,
			"background": true
		},
		{
			"type": "ellipse",
			"intensity": 0.7,
			"axis": [0.55, 0.55],
			"center": [0.1, 0.1],
			"angle": 10,
			"background": false
		},
		{
			"type": "ellipse",
			"intensity": 0.5,
			"axis": [0.25, 0.15],
			"center": [0.3, -0.2],
			"angle": 45,
			"background": false
		},
		{
			"type": "ellipse",
			"intensity": 0.3,
			"axis": [0.3, 0.05],
			"center": [0, 0],
			"angle": -45,
			"background": false
		}
	]
}
*/




static const char usage_str[] = "<geometry.json> <output>";
static const char help_str[] = "Image and k-space domain phantoms.";

int main_phantom_json(int argc, char* argv[])
{
	bool kspace = false;
	int sens = 0;
	int osens = -1;
	bool sens_out = false;
	int xdim = -1;


	const char* traj = NULL;
	bool base = false;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 128;
	dims[1] = 128;
	dims[2] = 1;
	
	
	const struct opt_s opts[] = {

		OPT_INT('s', &sens, "nc", "nc sensitivities"),
		OPT_INT('S', &osens, "nc", "Output nc sensitivities"),
		OPT_SET('k', &kspace, "k-space"),
		OPT_STRING('t', &traj, "file", "trajectory"),
		OPT_INT('x', &xdim, "n", "dimensions in y and z"),
		OPT_SET('b', &base, "create basis geometry"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	
	if (-1 != osens) {
		sens = osens;
		sens_out = true;
	}
	
	if (-1 != xdim)
		dims[0] = dims[1] = xdim;


	long sdims[DIMS];
	long sstrs[DIMS];
	complex float* samples = NULL;

	// Trajectory Input?
	if (NULL != traj) {

		if (-1 != xdim)
			debug_printf(DP_WARN, "size ignored.\n");

		kspace = true;

		samples = load_cfl(traj, DIMS, sdims);

		md_calc_strides(DIMS, sstrs, sdims, sizeof(complex float));

		dims[0] = 1;
		dims[1] = sdims[1];
		dims[2] = sdims[2];

		dims[TE_DIM] = sdims[TE_DIM];
	}
	
	
	// Import json geometry to ellipsis_e struct
	char* file = readfile(argv[1]);
	
	cJSON* json_data = cJSON_Parse(file);
	check_json_file(json_data);
	
	// Get array from json file
	const cJSON* objects = NULL;
	
	objects = cJSON_GetObjectItemCaseSensitive(json_data, "objects");

	int N = cJSON_GetArraySize(objects);
	
	struct ellipsis_s phantom_data[N];
	
	read_json_to_struct(N, phantom_data, objects);
	
	cJSON_Delete(json_data);
	
	
	// if base set coeff to length of ellipsis_s struct
	if (base)
		dims[COEFF_DIM] = N; 
		
	if (sens > 0)
		dims[3] = sens;

	
	// Initalize output
	complex float* out;
	out = create_cfl(argv[2], DIMS, dims);

	md_zfill(DIMS, dims, out, 0.);
	md_clear(DIMS, dims, out, sizeof(complex float));
	
	
	// Main
	if (sens_out) {

		assert(NULL == traj);
		assert(!kspace);

		calc_sens(dims, out);
	}
	else 
		(base ? calc_phantom_arb_base : calc_phantom_arb)(N, phantom_data, dims, out, kspace, sstrs, samples);

	
	// Clean up
	if (NULL != traj)
		free((void*)traj);
	
	if (NULL != samples)
		unmap_cfl(3, sdims, samples);
	
	unmap_cfl(DIMS, dims, out);
	
	return 0;
}


