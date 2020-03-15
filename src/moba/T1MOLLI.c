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
#include "T1relax_so.h"
#include "T1srelax.h"

//#define general
//#define mphase

struct nlop_s* nlop_T1MOLLI_create(int N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], const complex float* TI1, const complex float* TI2, bool use_gpu) 
{

#ifdef USE_CUDA
        md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
        assert(!use_gpu);
        md_alloc_fun_t my_alloc = md_alloc;
#endif

        long scale_dims[N];
        long TI_T1_dims[N];

        md_singleton_dims(N, scale_dims);
        md_singleton_dims(N, TI_T1_dims);

        complex float* scale = my_alloc(N, scale_dims, CFL_SIZE);
        complex float* TI_T1 = my_alloc(N, TI_T1_dims, CFL_SIZE);

        complex float TI_T1_init[1] = {0.0};
        md_copy(N, TI_T1_dims, TI_T1, TI_T1_init, CFL_SIZE);

        long out2_dims[N];
        md_copy_dims(N, out2_dims, out_dims);
        out2_dims[TE_DIM] = TI_T1_dims[TE_DIM];


        struct nlop_s* T1s_1 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI1);
        struct nlop_s* T1_1 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI_T1_dims, TI_T1);
        struct nlop_s* T1s_2 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI2);
        struct nlop_s* T1_2 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI_T1_dims, TI_T1);
        struct nlop_s* T1s_3 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI2);
        
        // first chain: T1(T1s)
        struct nlop_s* T1c_combine = nlop_combine(T1_1, T1s_1);
        struct nlop_s* T1c_link = nlop_link(T1c_combine, 2, 0);
        
        struct nlop_s* T1c_dup1 = nlop_dup(T1c_link, 1, 4);
        struct nlop_s* T1c_dup1_1 = nlop_dup(T1c_dup1, 0, 3);

        nlop_free(T1s_1);
        nlop_free(T1_1);
        nlop_free(T1c_combine);

        // second chain T1s(T1(T1s))
        struct nlop_s* T1c_combine2 = nlop_combine(T1s_2, T1c_dup1_1);
        struct nlop_s* T1c_link2 = nlop_link(T1c_combine2, 2, 0);

        struct nlop_s* T1c_dup2 = nlop_dup(T1c_link2, 2, 6);
        struct nlop_s* T1c_dup2_1 = nlop_dup(T1c_dup2, 1, 4);
        struct nlop_s* T1c_dup2_2 = nlop_dup(T1c_dup2_1, 0, 3);

        // struct nlop_s* T1c_dup2_2_del = nlop_del_out(T1c_dup2_2, 1);

        nlop_free(T1c_link);
        nlop_free(T1s_2);
        nlop_free(T1c_combine2);

        // third chain: T1(T1s(T1(T1s)))
        struct nlop_s* T1c_combine3 = nlop_combine(T1_2, T1c_dup2_2);
        struct nlop_s* T1c_link3 = nlop_link(T1c_combine3, 2, 0);

        struct nlop_s* T1c_dup3 = nlop_dup(T1c_link3, 1, 3);
        struct nlop_s* T1c_dup3_1 = nlop_dup(T1c_dup3, 0, 2);

        nlop_free(T1c_combine3);
        nlop_free(T1_2);
        nlop_free(T1c_dup2_2);

        //  stack the outputs together
        long sodims[N];
        md_copy_dims(N, sodims, out_dims);
        sodims[TE_DIM] = 2 * out_dims[TE_DIM];
        struct nlop_s* stack = nlop_stack_create(N, sodims, out_dims, out_dims, TE_DIM);

        struct nlop_s* T1c_combine_stack = nlop_combine(stack, T1c_dup3_1);
        struct nlop_s* T1c_link_stack_1 = nlop_link(T1c_combine_stack, 3, 0);

        struct nlop_s* T1c_link_stack_2 = nlop_link(T1c_link_stack_1, 2, 0);

        
        nlop_free(stack);
        nlop_free(T1c_combine_stack);
        nlop_free(T1c_dup3_1);
        nlop_free(T1c_link_stack_1); 

        // fourth chain : T1s(T1(T1s(T1(T1s))))
        struct nlop_s* T1c_combine4 = nlop_combine(T1s_3, T1c_link_stack_2);
        struct nlop_s* T1c_link4 = nlop_link(T1c_combine4, 3, 0);

        struct nlop_s* T1c_dup4 = nlop_dup(T1c_link4, 2, 5);
        struct nlop_s* T1c_dup4_1 = nlop_dup(T1c_dup4, 1, 4);
        struct nlop_s* T1c_dup4_2 = nlop_dup(T1c_dup4_1, 0, 3);

        nlop_free(T1c_combine4);
        nlop_free(T1s_3);
        nlop_free(T1c_link_stack_2);

          //  stack the outputs together
        long sodims1[N];
        md_copy_dims(N, sodims1, out_dims);
        sodims1[TE_DIM] = sodims[TE_DIM] + out_dims[TE_DIM];
        struct nlop_s* stack1 = nlop_stack_create(N, sodims1, sodims, out_dims, TE_DIM);

        struct nlop_s* T1c_combine_stack1 = nlop_combine(stack1, T1c_dup4_2);
        struct nlop_s* T1c_link_stack_1_1 = nlop_link(T1c_combine_stack1, 3, 0);

        struct nlop_s* T1c_link_stack_2_1 = nlop_link(T1c_link_stack_1_1, 1, 0);

        
        nlop_free(stack1);
        nlop_free(T1c_combine_stack1);
        nlop_free(T1c_dup4_2);
        nlop_free(T1c_link_stack_1_1); 

        struct nlop_s* T1c_del = nlop_del_out(T1c_link_stack_2_1, 1);

        // // scaling operator
        complex float diag[1] = {-1.0};
        md_copy(N, scale_dims, scale, diag, CFL_SIZE);

        struct linop_s* linop_scalar = linop_cdiag_create(N, map_dims, COEFF_FLAG, scale);

        struct nlop_s* nl_scalar = nlop_from_linop(linop_scalar);

        linop_free(linop_scalar);

        struct nlop_s* T1c_combine_scale = nlop_combine(T1c_del, nl_scalar);
        struct nlop_s* T1c_link_scale = nlop_link(T1c_combine_scale, 1, 3);
        struct nlop_s* T1c_dup_scale = nlop_dup(T1c_link_scale, 0, 3);

        nlop_free(nl_scalar);
        nlop_free(T1c_del);
        nlop_free(T1c_combine_scale);


        md_free(scale);
        md_free(TI_T1);


        return T1c_dup_scale;
}