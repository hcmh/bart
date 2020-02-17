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

//#define general
//#define mphase

struct nlop_s* nlop_T1MOLLI_create(int N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], const complex float* TI, bool use_gpu) 
{

        #ifdef USE_CUDA
        md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
        #else
        assert(!use_gpu);
        md_alloc_fun_t my_alloc = md_alloc;
        #endif

        long scale_dims[N];
        long TI2_dims[N];

        md_singleton_dims(N, scale_dims);
        md_singleton_dims(N, TI2_dims);

        complex float* scale = my_alloc(N, scale_dims, CFL_SIZE);
        complex float* TI2 = my_alloc(N, TI2_dims, CFL_SIZE);

        long out2_dims[N];
        md_copy_dims(N, out2_dims, out_dims);
        out2_dims[TE_DIM] = 1;


        struct nlop_s* T1s_1 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI);
        struct nlop_s* T1_1 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI2_dims, TI2);
        struct nlop_s* T1s_2 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI);
        
        // first chain
        struct nlop_s* T1c_combine = nlop_combine(T1_1, T1s_1);
        struct nlop_s* T1c_link = nlop_link(T1c_combine, 2, 0);
        
        struct nlop_s* T1c_dup1 = nlop_dup(T1c_link, 1, 4);
        struct nlop_s* T1c_dup1_1 = nlop_dup(T1c_dup1, 0, 3);

        nlop_free(T1s_1);
        nlop_free(T1_1);
        nlop_free(T1c_combine);

        // second chain
        struct nlop_s* T1c_combine2 = nlop_combine(T1s_2, T1c_dup1_1);
        struct nlop_s* T1c_link2 = nlop_link(T1c_combine2, 2, 0);

        struct nlop_s* T1c_dup2 = nlop_dup(T1c_link2, 2, 6);
        struct nlop_s* T1c_dup2_1 = nlop_dup(T1c_dup2, 1, 4);
        struct nlop_s* T1c_dup2_2 = nlop_dup(T1c_dup2_1, 0, 3);

        // struct nlop_s* T1c_dup2_2_del = nlop_del_out(T1c_dup2_2, 1);

        nlop_free(T1c_link);
        nlop_free(T1c_dup1_1);
        nlop_free(T1s_2);
        nlop_free(T1c_combine2);

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
        md_free(TI2);


        return T1s_dup_scale;
}