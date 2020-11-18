
#include "misc/opts.h"

#include "seq/seq.h"

static const char usage_str[] = "<outfile>";
static const char help_str[] = "Computes sequence.";


int main_seq(int argc, char* argv[])
{
	const struct opt_s opts[] = {
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	struct seq_flash_conf fl = seq_flash_defaults;

	struct seq_event ev[12];
	seq_flash(12, ev, fl, &seq_sys_skyra);

	float g[1000][3];
	seq_compute_gradients(1000, g, 2. * fl.TR / 1000., 12, ev);

	for (int i = 0; i < 1000; i++)
		printf("%e %e %e\n", g[i][0], g[i][1], g[i][2]);

	return 0;
}



