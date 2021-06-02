
#ifndef __OPTREG_H
#define __OPTREG_H

#include "misc/cppwrap.h"

#define NUM_REGS 10

struct operator_p_s;
struct linop_s;

struct reg_s
{
    enum {L1WAV, L2IMG, TV, LOGP, LOGPA} xform;

    unsigned int xflags;
	unsigned int jflags;

    float lambda;
	unsigned int k;

	char *graph_file;
	double pct;	
	unsigned int steps;
    unsigned int prior_dimx;
    unsigned int prior_dimy;

    unsigned int irgnm_steps;
    float base;
    float rho;
};

struct opt_reg_s
{
    float lambda;
    struct reg_s regs[NUM_REGS];
    unsigned int r;
    unsigned int k;
};

extern _Bool opt_reg_nlinv_init(struct opt_reg_s* ropts);

extern void opt_reg_nlinv_configure(unsigned int N, const long img_dims[__VLA(N)], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int shift_mode);

extern void opt_reg_nlinv_free(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS]);

extern _Bool opt_reg_nlinv(void* ptr, char c, const char* optarg);

extern void help_reg_nlinv(void);

#endif