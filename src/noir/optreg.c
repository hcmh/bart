
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/iovec.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "iter/prox.h"
#include "iter/prox2.h"
#include "iter/thresh.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/waveop.h"
#include "linops/fmac.h"

#include "wavelet/wavthresh.h"

#include "lowrank/lrthresh.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"

#include "nn/tf_wrapper.h"

#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "optreg.h"

#define CFL_SIZE sizeof(complex float)

void help_reg_nlinv(void)
{
	printf( "Generalized regularization options (experimental)\n\n"
			"-R <T>:A:B:C\t<T> is regularization type (single letter),\n"
			"\t\tA is transform flags, B is joint threshold flags,\n"
			"\t\tand C is regularization value. Specify any number\n"
			"\t\tof regularization terms.\n\n"
			"-R W:A:B:C\tl1-wavelet\n"
			"-R T:A:B:C\ttotal variation\n"
			"-R LP:{graph_path}:lambda:steps\t PixelCNN as image regularization\n"
			"-R DP:{graph_path}\t diffusion prior as image regularization\n"
	      );    
}

bool opt_reg_nlinv(void* ptr, char c, const char* optarg)
{
    struct opt_reg_s* p = ptr;
    struct reg_s* regs = p->regs;
    const int r = p->r;
    const float lambda = p->lambda;

    assert(r < NUM_REGS);

    char rt[5];

    switch (c) {
        
    case 'R':{
        // first get transform type
        int ret = sscanf(optarg, "%4[^:]", rt);
        assert(1 == ret);
        
        // next switch based on transform type
        if (strcmp(rt, "W") == 0){

            regs[r].xform = L1WAV;
            int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
        }
        else if (strcmp(rt, "T") == 0)
        {
            int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
        }
        else if (strcmp(rt, "LP") == 0)
		{
			
			regs[r].xform = LOGP;
			regs[r].graph_file = (char *)malloc(100*sizeof(char));
			int ret = sscanf(optarg, "%*[^:]:{%[^}]}:%f:%u", regs[r].graph_file, &regs[r].lambda, &regs[r].steps);
			assert(3 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "DP") == 0)
		{
			regs[r].xform = LOGDP;
			regs[r].graph_file = (char *)malloc(100*sizeof(char));
			int ret = sscanf(optarg, "%*[^:]:{%[^}]}:%f:%u", regs[r].graph_file, &regs[r].lambda, &regs[r].k);
			assert(3 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "h") == 0) {

			help_reg_nlinv();
			exit(0);
		}
        else {

			error("Unrecognized regularization type: \"%s\" (-Rh for help).\n", rt);
		}

		p->r++;
		break;
        
    }

    case 'l':{

		assert(r < NUM_REGS);
		regs[r].lambda = lambda;
		regs[r].xflags = 0u;
		regs[r].jflags = 0u;

		if (0 == strcmp("1", optarg)) {

			regs[r].xform = L1WAV;
			regs[r].xflags = FFT_FLAGS; //

		} else if (0 == strcmp("2", optarg)) {

			regs[r].xform = L2IMG;

		} else {

			error("Unknown regularization type.\n");
		}

		p->lambda = -1.;
		p->r++;
		break;
	}
    }
	return false;
}

bool opt_reg_nlinv_init(struct opt_reg_s* ropts)
{
	ropts->r = 0;
	ropts->lambda = -1;
	ropts->k = 0;

	return false;
}


static const struct operator_p_s* nlinv_sens_prox_create(unsigned int N, const long sens_dims[N])
{
	const struct operator_p_s* prox = prox_zero_create(N, sens_dims);
	return prox;
}

static const struct operator_p_s* flatten_prox(const struct operator_p_s* src)
{
	const struct operator_p_s* dst = operator_p_reshape_in_F(src, 1, MD_DIMS(md_calc_size(operator_p_domain(src)->N, operator_p_domain(src)->dims)));
	dst = operator_p_reshape_out_F(dst, 1, MD_DIMS(md_calc_size(operator_p_codomain(dst)->N, operator_p_codomain(dst)->dims)));
	return dst;
}

static const struct operator_p_s* stack_flatten_prox(const struct operator_p_s* prox_maps, const struct operator_p_s* prox_sens)
{
	auto prox1 = flatten_prox(prox_maps);
	auto prox2 = flatten_prox(prox_sens);
	auto prox3 = operator_p_stack(0, 0, prox1, prox2);
	operator_p_free(prox1);
	operator_p_free(prox2);
	return prox3;
}



void opt_reg_nlinv_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int shift_mode, const char* wtype_str)
{

	bool randshift = shift_mode == 1;
	
    float lambda = ropts->lambda;

	long img_dims[N];
	md_select_dims(N, ~COIL_FLAG, img_dims, dims);

	long coil_dims[N];
	md_select_dims(N, ~COEFF_FLAG, coil_dims, dims);
	
	long x_dims[DIMS];
	md_copy_dims(DIMS, x_dims, coil_dims);
	x_dims[COIL_DIM] = x_dims[COIL_DIM] + 1;

    if (-1. == lambda)
		lambda = 0.;

    // if no penalities specified but regularization
	// parameter is given, add a l2 penalty

	struct reg_s* regs = ropts->regs;

	if ((0 == ropts->r) && (lambda > 0.)) {

		regs[0].xform = L2IMG;
		regs[0].xflags = 0u;
		regs[0].jflags = 0u;
		regs[0].lambda = lambda;
		ropts->r = 1;
	}

	enum wtype wtype;

	if	(0 == strcmp("haar", wtype_str))
		wtype = WAVELET_HAAR;
	else if (0 == strcmp("dau2", wtype_str))
		wtype = WAVELET_DAU2;
	else if (0 == strcmp("cdf44", wtype_str))
		wtype = WAVELET_CDF44;
	else
		error("unsupported wavelet type.\n");

    for (int nr=0; nr < ropts->r; nr++){

        // fix up regularization parameter
        if (-1. == regs[nr].lambda)
            regs[nr].lambda = lambda;

        switch (regs[nr].xform)
        {

        case L1WAV:
        {
            debug_printf(DP_INFO, "l1-wavelet regularization: %f\n", regs[nr].lambda);

			long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
			minsize[0] = MIN(img_dims[0], 16);
			minsize[1] = MIN(img_dims[1], 16);
			minsize[2] = MIN(img_dims[2], 16);


			unsigned int wflags = 0;
			for (unsigned int i = 0; i < DIMS; i++) {

				if ((1 < img_dims[i]) && MD_IS_SET(regs[nr].xflags, i)) {

					wflags = MD_SET(wflags, i);
					minsize[i] = MIN(img_dims[i], 16);
				}
			}

			trafos[nr] = linop_identity_create(DIMS, x_dims);

			auto prox_img = prox_wavelet_thresh_create(DIMS, img_dims, wflags, regs[nr].jflags, wtype, minsize, regs[nr].lambda, randshift); 

			auto prox_coil = nlinv_sens_prox_create(DIMS, coil_dims);

			prox_ops[nr] = stack_flatten_prox(prox_img, prox_coil);
			break;
		}

		case TV:
		{
			debug_printf(DP_INFO, "TV regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_grad_create(DIMS, img_dims, DIMS, regs[nr].xflags);
			prox_ops[nr] = prox_thresh_create(DIMS + 1,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS));
			break;
		}
		case L2IMG:
		{
			debug_printf(DP_INFO, "l2 regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_identity_create(DIMS, x_dims);
			
			auto prox_img = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);

			auto prox_coil = nlinv_sens_prox_create(DIMS, coil_dims);

			prox_ops[nr] = stack_flatten_prox(prox_img, prox_coil);
			break;
		}
		
		case LOGP:
		{
			debug_printf(DP_INFO, "PixelCNN as image regularization. lambda: %f steps: %u\n", regs[nr].lambda, regs[nr].steps);
			
			trafos[nr] = linop_identity_create(DIMS, x_dims);

			const struct nlop_s * tf_ops = nlop_tf_create(regs[nr].graph_file);

			auto prox_op = prox_nlgrad_create(tf_ops, regs[nr].steps, 1., regs[nr].lambda);

			auto prox_coil = nlinv_sens_prox_create(DIMS, coil_dims);

			auto prox_img = op_p_auto_normalize(prox_op, ~0LU, NORM_MAX);

			prox_ops[nr] = stack_flatten_prox(prox_img, prox_coil);
			break;
		}

		case LOGDP:
		{
			debug_printf(DP_INFO, "Diffusion prior as image regularization\n");

			const struct tf_shared_graph_s* tf_graph = tf_shared_graph_create(regs[nr].graph_file, NULL);
			tf_shared_graph_set_batch_size(tf_graph, img_dims[READ_DIM]);

			const struct nlop_s* nlop = nlop_tf_shared_create(tf_graph);

			auto prox_op = prox_nl_dp_grad_create(nlop, regs[nr].k, regs[nr].lambda); 

			auto prox_coil = nlinv_sens_prox_create(DIMS, coil_dims);

			prox_ops[nr] = stack_flatten_prox(prox_op, prox_coil);
			break;
		}
        }        
    }
}

void opt_reg_nlinv_free(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS])
{

	for (int nr = 0; nr < ropts->r; nr++) {

		operator_p_free(prox_ops[nr]);
		linop_free(trafos[nr]);
	}
}