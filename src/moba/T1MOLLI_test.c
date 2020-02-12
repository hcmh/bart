#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "T1MOLLI.h"
#include "T1relax.h"
#include "T1srelax.h"

//#define general
//#define mphase

const struct nlop_s* testMOLLI(long N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], complex float* TI) 
{

        struct nlop_s* T1s_1 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI);
    	struct nlop_s* T1s_2 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI);
   
    	struct nlop_s* T1s_combine = nlop_combine(T1s_2, T1s_1);
        struct nlop_s* T1s_link = nlop_link(T1s_combine, 3, 0);
    	struct nlop_s* T1s_dup1 = nlop_dup(T1s_link, 2, 6);
   	struct nlop_s* T1s_dup2 = nlop_dup(T1s_dup1, 1, 5);
        struct nlop_s* T1s_dup3 = nlop_dup(T1s_dup2, 0, 4);

        long sodims[N];
        md_copy_dims(N, sodims, out_dims);
        sodims[TE_DIM] = 2 * out_dims[TE_DIM];
        struct nlop_s* stack = nlop_stack_create(N, sodims, out_dims, out_dims, TE_DIM);

        struct nlop_s* T1s_combine_stack = nlop_combine(stack, T1s_dup3);
        struct nlop_s* T1s_link_stack_1 = nlop_link(T1s_combine_stack, 1, 1);

        struct nlop_s* T1s_link_stack_2 = nlop_link(T1s_link_stack_1, 2, 0);

    	
        nlop_free(T1s_combine_stack);
        nlop_free(T1s_link_stack_1);
        nlop_free(T1s_1);
    	nlop_free(T1s_2);
    	nlop_free(T1s_combine);


        return T1s_link_stack_2;
}