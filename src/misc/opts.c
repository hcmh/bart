/* Copyright 2015-2017. Martin Uecker.
 * Copyright 2017-2018. Damien Nguyen.
 * Copyright 2017-2018. Francesco Santini.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2017-2018 Nguyen Damien <damien.nguyen@alumni.epfl.ch>
 * 2017-2018 Francesco Santini <francesco.santini@unibas.ch>
 */

#define _GNU_SOURCE
#include "ya_getopt.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <complex.h>


#include "misc/misc.h"
#include "misc/debug.h"

#include "opts.h"

enum OPT_ARG_TYPE { OPT_SPECIAL, OPT_SET, OPT_CLEAR, OPT_INT, OPT_UINT, OPT_LONG, OPT_FLOAT, OPT_CFLOAT, OPT_STRING };

static const char* opt_arg_types[] = { " ...", "", "", " d", " d", " d", " f", " cf", " <string>" };
static const char* opt_arg_types_subopts[] = { "=...", "", "", "=d", "=d", "=d", "=f", "=cf", "=<string>" };

static enum OPT_ARG_TYPE opt_arg_type(opt_conv_f fun)
{
	if (opt_set == fun)
		return OPT_SET;

	if (opt_clear == fun)
		return OPT_CLEAR;

	if (opt_int == fun)
		return OPT_INT;

	if (opt_uint == fun)
		return OPT_UINT;

	if (opt_long == fun)
		return OPT_LONG;

	if (opt_float == fun)
		return OPT_FLOAT;

	if (opt_cfloat == fun)
		return OPT_CFLOAT;

	if (opt_string == fun)
		return OPT_STRING;

	return OPT_SPECIAL;
}

static const char* trim_space(const char* str)
{
	while (isspace(*str))
		str++;

	return str;
}

static bool show_option_p(const struct opt_s opt)
{
	return     (NULL != opt.descr)
		&& !(   ('(' == trim_space(opt.descr)[0])
		     && (')' == opt.descr[strlen(opt.descr) - 1]));
}

static void print_usage(FILE* fp, const char* name, const char* usage_str, int n, const struct opt_s opts[static n ?: 1])
{
	fprintf(fp, "Usage: %s ", name);

	for (int i = 0; i < n; i++)
		if (show_option_p(opts[i])) {

			if (NULL == opts[i].s) {

				fprintf(fp, "[-%c%s] ", opts[i].c, opt_arg_types[opt_arg_type(opts[i].conv)]);
			} else {

				if (opts[i].c < (int) ' ')
					fprintf(fp, "[--%s%s] ", opts[i].s, opt_arg_types[opt_arg_type(opts[i].conv)]);
				else
					fprintf(fp, "[-%c,--%s%s] ", opts[i].c, opts[i].s, opt_arg_types[opt_arg_type(opts[i].conv)]);
			}
		}

	fprintf(fp, "%s\n", usage_str);
}

static void print_usage_subopts(FILE* fp, char c, const char* arg_name, const char* usage_str, int n, const struct opt_s opts[static n ?: 1])
{
	fprintf(fp, "Usage of sub-option: ");

	if (0 != c)
		fprintf(fp, "-%c", c);

	if (NULL != arg_name)
		fprintf(fp, (0 == c) ? "--%s " : ",--%s ", arg_name);

	for (int i = 0; i < n; i++)
		if (show_option_p(opts[i])) {

			if (NULL == opts[i].s) {

				fprintf(fp, "[%c%s]%s", opts[i].c, opt_arg_types_subopts[opt_arg_type(opts[i].conv)], i < (n - 1) ? "," : "");
			} else {

				if (opts[i].c < (int) ' ')
					fprintf(fp, "[%s%s]%s", opts[i].s, opt_arg_types_subopts[opt_arg_type(opts[i].conv)], i < (n - 1) ? "," : "");
				else
					fprintf(fp, "[%c%s,%s%s]%s", opts[i].c, opt_arg_types_subopts[opt_arg_type(opts[i].conv)], opts[i].s, opt_arg_types_subopts[opt_arg_type(opts[i].conv)], i < (n - 1) ? "," : "");
			}
		}

	fprintf(fp, "%s\n", usage_str);
}


static const char* add_space(bool has_arg, bool has_space)
{
	const char* space = "\t\t";

	if (has_arg)
		space = " ";

	if (!has_space)
		space = "";

	return space;
}


static void print_help(const char* help_str, int n, const struct opt_s opts[n ?: 1])
{
	printf("\n%s\n\n",  help_str);

	for (int i = 0; i < n; i++)
		if (show_option_p(opts[i])) {

			if (NULL == opts[i].s) {


				printf("-%c%s%s\n", opts[i].c,
					add_space(opts[i].arg, isspace(opts[i].descr[0])),
					trim_space(opts[i].descr));
			} else {
				if (opts[i].c < (int) ' ')
					printf("--%s%s%s\n", opts[i].s,
						add_space(opts[i].arg, isspace(opts[i].descr[0])),
						trim_space(opts[i].descr));
				else
					printf("-%c,--%s%s%s\n", opts[i].c, opts[i].s,
					       add_space(opts[i].arg, isspace(opts[i].descr[0])),
					       trim_space(opts[i].descr));
			}
		}


	printf("-h\t\thelp\n");
}

static void print_help_subopts(const char* descr, int n, const struct opt_s opts[n ?: 1])
{

	if (NULL != descr)
		printf("\nSub-options: %s\n\n",  descr);
	else
		printf("\n");

	for (int i = 0; i < n; i++)
		if (show_option_p(opts[i])) {

			if (NULL == opts[i].s) {


				printf("%c%s%s\n", opts[i].c,
					add_space(opts[i].arg, isspace(opts[i].descr[0])),
					trim_space(opts[i].descr));
			} else {
				if (opts[i].c < (int) ' ')
					printf("%s%s%s\n", opts[i].s,
						add_space(opts[i].arg, isspace(opts[i].descr[0])),
						trim_space(opts[i].descr));
				else
					printf("%c,%s%s%s\n", opts[i].c, opts[i].s,
					       add_space(opts[i].arg, isspace(opts[i].descr[0])),
					       trim_space(opts[i].descr));
			}
		}

	printf("h\t\thelp\n");
}




static void check_options(int n, const struct opt_s opts[n ?: 1])
{
	bool f[256] = { false };

	for (int i = 0; i < n; i++) {

		int c = opts[i].c;

		assert(256 > c);

		if (f[c])
			error("duplicate option: %c\n", opts[i].c);

		f[c] = true;
	}
}


static void process_option(char c, const char* optarg, const char* name, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n ?: 1])
{
	if ('h' == c) {

		print_usage(stdout, name, usage_str, n, opts);
		print_help(help_str, n, opts);
		exit(0);
	}

	for (int i = 0; i < n; i++) {

		if (opts[i].c == c) {

			if (opts[i].conv(opts[i].ptr, c, optarg)) {

				print_usage(stderr, name, usage_str, n, opts);
				error("process_option: failed to convert value\n");
			}

			return;
		}
	}

	print_usage(stderr, name, usage_str, n, opts);
	error("process_option: unknown option\n");
}


void cmdline(int* argcp, char* argv[], int min_args, int max_args, const char* usage_str, const char* help_str, int n, const struct opt_s opts[n ?: 1])
{
	int argc = *argcp;

	// create writable copy of opts

	struct opt_s wopts[n ?: 1];

	if (NULL != opts)
		memcpy(wopts, opts, sizeof wopts);


	int max_num_long_opts = ' ';
	struct option longopts[max_num_long_opts];

	// According to documentation, the last element of the longopts array has to be filled with zeros.
	// So we fill it entirely before using it

	memset(longopts, 0, sizeof longopts);


	char lc = 1;
	int nlong = 0;
	for (int i = 0; i < n; i++) {

		if (NULL != wopts[i].s) {

			// if it is only longopt, overwrite c with an unprintable char
			if (0 == wopts[i].c)
				wopts[i].c = lc++;

			longopts[nlong++] = (struct option){ wopts[i].s, wopts[i].arg, NULL, wopts[i].c };
		}
	}

	// Ensure that we only used unprintable characters
	// and that the last entry of the array is only zeros

	assert(nlong < max_num_long_opts);

#if 0
	for (int i = 0; i < n; ++i)
		debug_printf(DP_INFO, "opt: %d: %s: %d\n",i, wopts[i].descr, (int) wopts[i].c);
#endif



	char optstr[2 * n + 2];
	ya_getopt_reset(); // reset getopt variables to process multiple argc/argv pairs

	check_options(n, wopts);

	save_command_line(argc, argv);

	int l = 0;
	optstr[l++] = 'h';

	for (int i = 0; i < n; i++) {

		optstr[l++] = wopts[i].c;

		if (wopts[i].arg)
			optstr[l++] = ':';
	}

	optstr[l] = '\0';

	int c;
	int longindex = -1;

	while (-1 != (c = ya_getopt_long(argc, argv, optstr, longopts, &longindex))) {
	//while (-1 != (c = ya_getopt(argc, argv, optstr))) {
#if 0
		if ('h' == c) {

			print_usage(stdout, argv[0], usage_str, n, opts);
			print_help(help_str, n, opts);
			exit(0);
		}

		for (int i = 0; i < n; i++) {

			if (opts[i].c == c) {

				if (opts[i].conv(opts[i].ptr, c, optarg)) {

					print_usage(stderr, argv[0], usage_str, n, opts);
					error("cmdline: failed to convert value\n");
				}

				goto out;
			}
		}

		print_usage(stderr, argv[0], usage_str, n, opts);
		exit(1);

	out:	continue;
#else
		process_option(c, optarg, argv[0], usage_str, help_str, n, wopts);
#endif
	}

	if (   (argc - optind < min_args)
	    || (argc - optind > max_args)) {

		print_usage(stderr, argv[0], usage_str, n, wopts);
		error("cmdline: too few or too many arguments\n");
	}

	int i;
	for (i = optind; i < argc; i++)
		argv[i - optind + 1] = argv[i];

	*argcp = argc - optind + 1;
	argv[*argcp] = NULL;
}


bool opt_set(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(bool*)ptr = true;
	return false;
}

bool opt_clear(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	*(bool*)ptr = false;
	return false;
}

bool opt_int(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(int*)ptr = atoi(optarg);
	return false;
}

bool opt_uint(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(unsigned int*)ptr = atoi(optarg);
	return false;
}

bool opt_long(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(long*)ptr = atoi(optarg);
	return false;
}

bool opt_float(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(float*)ptr = atof(optarg);
	return false;
}

bool opt_cfloat(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	parse_cfl((complex float*)ptr, optarg);
	return false;
}

bool opt_string(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	*(char**)ptr = strdup(optarg);
	assert(NULL != ptr);
	return false;
}

bool opt_float_vec2(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	int r = sscanf(optarg, "%f:%f", &(*(float(*)[2])ptr)[0], &(*(float(*)[2])ptr)[1]);

	assert(2 == r);

	return false;
}

bool opt_vec2(void* ptr, char c, const char* optarg)
{
	if (islower(c)) {

		if (2 != sscanf(optarg, "%ld:%ld", &(*(long(*)[2])ptr)[0], &(*(long(*)[2])ptr)[1])) {

			(*(long(*)[3])ptr)[0] = atol(optarg);
			(*(long(*)[3])ptr)[1] = atol(optarg);
		}

	} else {

		debug_printf(DP_WARN, "the upper-case options for specifying dimensions are deprecated.\n");
		int r = sscanf(optarg, "%ld:%ld", &(*(long(*)[2])ptr)[0], &(*(long(*)[2])ptr)[1]);
		assert(2 == r);
	}

	return false;
}

bool opt_float_vec3(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	int r = sscanf(optarg, "%f:%f:%f", &(*(float(*)[3])ptr)[0], &(*(float(*)[3])ptr)[1], &(*(float(*)[3])ptr)[2]);

	assert(3 == r);

	return false;
}


bool opt_vec3(void* ptr, char c, const char* optarg)
{
	if (islower(c)) {

		if (3 != sscanf(optarg, "%ld:%ld:%ld", &(*(long(*)[3])ptr)[0], &(*(long(*)[3])ptr)[1], &(*(long(*)[3])ptr)[2])) {

			(*(long(*)[3])ptr)[0] = atol(optarg);
			(*(long(*)[3])ptr)[1] = atol(optarg);
			(*(long(*)[3])ptr)[2] = atol(optarg);
		}

	} else {

		debug_printf(DP_WARN, "the upper-case options for specifying dimensions are deprecated.\n");
		int r = sscanf(optarg, "%ld:%ld:%ld", &(*(long(*)[3])ptr)[0], &(*(long(*)[3])ptr)[1], &(*(long(*)[3])ptr)[2]);
		assert(3 == r);
	}

	return false;
}

bool opt_select(void* ptr, char c, const char* optarg)
{
	UNUSED(c); UNUSED(optarg);
	struct opt_select_s* sel = ptr;

	if (0 != memcmp(sel->ptr, sel->default_value, sel->size))
		return true;

	memcpy(sel->ptr, sel->value, sel->size);
	return false;
}

bool opt_subopt(void* _ptr, char c, const char* optarg)
{
	UNUSED(c);
	struct opt_subopt_s* ptr = _ptr;

	int n = ptr->n;
	auto opts = ptr->opts;

	struct opt_s wopts[n ?: 1];

	if (NULL != opts)
		memcpy(wopts, opts, sizeof wopts);

	char lc = 1;
	for (int i = 0; i < n; i++) {

		if (NULL != wopts[i].s) {

			// if it is only longopt, overwrite c with an unprintable char
			if (0 == wopts[i].c)
				wopts[i].c = lc++;

			assert(lc < ' ');
		}
	}

	const char* tokens[2 * ptr->n + 2];
	for (int i = 0; i < ptr->n; i++) {

		tokens[2 * i] = ptr_printf("%c", wopts[i].c);
		if (NULL == wopts[i].s)
			tokens[2 * i + 1] = ptr_printf("char_only_token_%c", wopts[i].c);
		else
			tokens[2 * i + 1] = ptr_printf("%s", wopts[i].s);
	}

	tokens[2 * ptr->n] = ptr_printf("h");
	tokens[2 * ptr->n + 1] = NULL;


	char* tmpoptionp = strdup(optarg);
	char* option = tmpoptionp;
	char* value = NULL;

	int i = -1;
	while('\0' != *option) {

		i = getsubopt(&option, (char *const *)tokens, &value);
		if ((i == 2 * n) || (-1 == i)) {

			print_usage_subopts(stdout, ptr->calling_c, ptr->calling_s, "", n, opts);
			print_help_subopts(ptr->calling_desc, ptr->n, ptr->opts);
		}

		if (-1 == i)
			error("Sub-option could not be parsed: %s", value);

		if (i < 2 * n)
			process_option(wopts[i / 2].c, value, "", "", "", n, wopts);
		else
			exit(0);
	}

	xfree(tmpoptionp);

	for (int i = 0; i < 2 * n + 1; i++)
		xfree(tokens[i]);


	return false;
}


bool opt_idx(void* ptr, char c, const char* optarg)
{
	UNUSED(c);
	struct opt_idx_s* p = ptr;
	struct idx_s* idxs = p->idxs;
	const unsigned int r = p->r;

	int ret = sscanf(optarg, "%d:%d", &idxs[r].dim, &idxs[r].idx);
	assert(2 == ret);

	assert(r < 16);

	p->r++;
	return false;
}

bool opt_idx_init(struct opt_idx_s* iopts)
{
	iopts->r = 0;
	return false;
}



