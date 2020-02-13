#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nltest.h"
#include "nlops/stack.h"
#include "nlops/const.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "T1MOLLI.h"
#include "T1relax.h"
#include "T1srelax.h"
#include "T1MOLLI_test.h"

//#define general
//#define mphase

struct nlop_s* nlop_T1MOLLI_test_create(int N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], const complex float* TI) 
{

        long scale_dims[N];

        md_singleton_dims(N, scale_dims);

        complex float* scale = md_alloc(N, scale_dims, CFL_SIZE);

        struct nlop_s* T1s_1 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI);
    	struct nlop_s* T1s_2 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI);
   
    	struct nlop_s* T1s_combine = nlop_combine(T1s_2, T1s_1);
        struct nlop_s* T1s_link = nlop_link(T1s_combine, 3, 0);
    	struct nlop_s* T1s_dup1 = nlop_dup(T1s_link, 2, 6);
   	    struct nlop_s* T1s_dup2 = nlop_dup(T1s_dup1, 1, 5);
        struct nlop_s* T1s_dup3 = nlop_dup(T1s_dup2, 0, 4);

        // stack two outputs
        long sodims[N];
        md_copy_dims(N, sodims, out_dims);
        sodims[TE_DIM] = 2 * out_dims[TE_DIM];
        struct nlop_s* stack = nlop_stack_create(N, sodims, out_dims, out_dims, TE_DIM);

        struct nlop_s* T1s_combine_stack = nlop_combine(stack, T1s_dup3);
        struct nlop_s* T1s_link_stack_1 = nlop_link(T1s_combine_stack, 1, 1);

        struct nlop_s* T1s_link_stack_2 = nlop_link(T1s_link_stack_1, 2, 0);

        struct nlop_s* T1s_link_stack_3 = nlop_del_out(T1s_link_stack_2, 1);


        // scaling operator
        complex float diag[1] = {-1.0};
        md_copy(N, scale_dims, scale, diag, CFL_SIZE);

        struct linop_s* linop_scalar = linop_cdiag_create(N, map_dims, COEFF_FLAG, scale);

        struct nlop_s* nl_scalar = nlop_from_linop(linop_scalar);

        linop_free(linop_scalar);

        struct nlop_s* T1s_combine_scale = nlop_combine(T1s_link_stack_3, nl_scalar);
        struct nlop_s* T1s_link_scale = nlop_link(T1s_combine_scale, 1, 3);
        struct nlop_s* T1s_dup_scale = nlop_dup(T1s_link_scale, 0, 3);


        nlop_free(T1s_combine_scale);
        nlop_free(nl_scalar);
    	nlop_free(T1s_link_stack_3);
        nlop_free(T1s_link_stack_2);
        nlop_free(T1s_link_stack_1);
        nlop_free(T1s_combine_stack);
        nlop_free(T1s_1);
    	nlop_free(T1s_2);
    	nlop_free(T1s_combine);

        md_free(scale);


        return T1s_dup_scale;
}