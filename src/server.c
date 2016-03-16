
#include <assert.h>
#include <strings.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <complex.h>
#include <sys/socket.h>

#include "num/multind.h"

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/net.h"



static size_t xread(int fd, size_t si, void* data)
{
	size_t len = 0;

	while (si > len) {

		ssize_t rd = read(fd, data + len, si - len);

		if (0 == rd)	// EOF
			break;

		len += rd;
	}

	return len;
}

static int parse_header(unsigned int* N, long** dims, char header[static 4096])
{
	int errno = 1;

	size_t pos = 0;
	int delta;	

	if (1 != sscanf(header + pos, "Dimensions: %u\n%n", N, &delta))
		goto err;

	pos += delta;

	{
	long d[*N];

	if (1 != sscanf(header + pos, "Sizes: %ld%n", &d[0], &delta))
		goto err;

	pos += delta;

	for (unsigned int i = 1; i < *N; i++) {

		if (1 != sscanf(header + pos, "x%ld%n", &d[i], &delta))
			goto err;

		pos += delta;
	}


	*dims = xmalloc(*N * sizeof(long));
	for (unsigned int i = 0; i < *N; i++)
		(*dims)[i] = d[i];

	if (0 != sscanf(header + pos, "\n%n", &delta))
		goto err;

	pos += delta;

	printf("HERE\n");

	errno = 0;
	}
err:	return errno;
}


static const char usage_str[] = "<output>";
static const char help_str[] = "Wait for a connection on <port> and update file.";


int main_server(int argc, char* argv[])
{
	int port = 2121;
	int errno = 1;
	
	const struct opt_s opts[] = {

		OPT_INT('p', &port, "p", "port"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);


	int fd;
	if (-1 == (fd = listen_at(port)))
		error("listen.\n");

	debug_printf(DP_INFO, "Server started.\n");

	complex float* data = NULL;
	long* dims = NULL;

	do {

		int fd2 = -1;

		if (-1 == (fd2 = accept(fd, NULL, NULL)))
			error("accept\n");

		debug_printf(DP_INFO, "Connection!");

		size_t len = 4096;
		char buf[len];

		if (len != xread(fd2, len, buf))
			goto err1;

		debug_printf(DP_INFO, "%s\n", buf);

		unsigned int N = 0;
		if (0 != parse_header(&N, &dims, buf))
			goto err1;

		debug_printf(DP_INFO, "\n");
		debug_print_dims(DP_INFO, N, dims);

		data = create_cfl(argv[1], N, dims);

		size_t dsize = md_calc_size(N, dims) * sizeof(complex float);

		if (dsize != xread(fd2, dsize, data))
			goto err;

		errno = 0;
err:
		unmap_cfl(N, dims, data);

err1:
		if (errno > 0)
			debug_printf(DP_WARN, "error reading file\n");

		close(fd2);

	} while(false);

	debug_printf(DP_INFO, "Server done.\n");

	exit(errno);
}

