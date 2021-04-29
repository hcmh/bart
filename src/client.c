
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <complex.h>
#include <strings.h>

#include "num/multind.h"

#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/net.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif


static size_t xwrite(int fd, size_t len, const void* buf)
{
	return write(fd, buf, len);	
}


static const char help_str[] = "Send file to host on port.";


int main_client(int argc, char* argv[argc])
{
	const char* input_file = NULL;
	const char* host_port = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &input_file, "input"),
		ARG_STRING(true, &host_port, "host:port"),
	};

	int port = 2121;
	int errno = 1;
	
	const struct opt_s opts[] = { 
	};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long dims[DIMS];
	complex float* data = load_cfl(input_file, DIMS, dims);

	char host[128];
	if (2 != sscanf(host_port, "%128[^:]:%d", host, &port))
		goto err1;

	debug_printf(DP_INFO, "Sending %s to %s:%d\n", input_file, host, port);

	int fd;

	if (-1 == (fd = connect_to(host, port)))
		goto err1;

	debug_printf(DP_INFO, "Connected. Sending header...\n");

	char buf[4096];
	bzero(buf, sizeof(buf));

	size_t pos = 0;
	pos += snprintf(buf + pos, 4096 - pos, "Dimensions: %d\n", DIMS);
	pos += snprintf(buf + pos, 4096 - pos, "Sizes: %ld", dims[0]);

	for (unsigned int i = 1; i < DIMS; i++)
		pos += snprintf(buf + pos, 4096 - pos, "x%ld", dims[i]);
		
	pos += snprintf(buf + pos, 4096 - pos, "\n");

	xwrite(fd, sizeof(buf), buf);

	debug_printf(DP_INFO, "Sending data...\n");

	size_t len = md_calc_size(DIMS, dims) * sizeof(complex float);

	if (len != xwrite(fd, len, data))
		goto err2;

	debug_printf(DP_INFO, "Done.\n");

	errno = 0;

err2:	close(fd);

err1:	unmap_cfl(DIMS, dims, data);
	exit(errno);
}

