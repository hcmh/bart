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
#include "T1_alpha2.h"

//#define general
//#define mphase

struct nlop_s* nlop_T1MOLLI_create(int N, const long map_dims[N], const long out_dims[N], const long TI_dims[N], 
                const complex float* TI1, const complex float* TI2, const complex float* TI_t1relax, bool use_gpu) 
{

        long scale_dims[N];
        long TI_T1_dims[N];

        md_singleton_dims(N, scale_dims);
        md_singleton_dims(N, TI_T1_dims);

        complex float* scale = md_alloc(N, scale_dims, CFL_SIZE);

        long out2_dims[N];
        md_copy_dims(N, out2_dims, out_dims);
        out2_dims[TE_DIM] = 1L;

#if 0
        struct nlop_s* T1s_1 = nlop_T1_alpha2_create(N, map_dims, out_dims, TI_dims, TI1, use_gpu);
        struct nlop_s* T1s_2 = nlop_T1_alpha2_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
        struct nlop_s* T1s_3 = nlop_T1_alpha2_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
        struct nlop_s* T1s_4 = nlop_T1_alpha2_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
        struct nlop_s* T1s_5 = nlop_T1_alpha2_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
#endif
	struct nlop_s* T1s_1 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI1, use_gpu);
        struct nlop_s* T1_1 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI_T1_dims, &TI_t1relax[0], use_gpu);
	struct nlop_s* T1s_2 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
        struct nlop_s* T1_2 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI_T1_dims, &TI_t1relax[1], use_gpu);
	struct nlop_s* T1s_3 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
        struct nlop_s* T1_3 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI_T1_dims, &TI_t1relax[2], use_gpu);
	struct nlop_s* T1s_4 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
        struct nlop_s* T1_4 = nlop_T1relax_so_create(N, map_dims, out2_dims, TI_T1_dims, &TI_t1relax[3], use_gpu);
	struct nlop_s* T1s_5 = nlop_T1srelax_create(N, map_dims, out_dims, TI_dims, TI2, use_gpu);
        
        // first chain: T1(T1s)
        struct nlop_s* T1c_combine = nlop_combine_FF(T1_1, T1s_1);
        struct nlop_s* T1c_link = nlop_link_F(T1c_combine, 2, 0);   
        struct nlop_s* T1c_dup1 = nlop_dup_F(T1c_link, 1, 4);
        struct nlop_s* T1c_dup1_1 = nlop_dup_F(T1c_dup1, 0, 3);

        // second chain T1s(T1(T1s))
        struct nlop_s* T1c_combine2 = nlop_combine_FF(T1s_2, T1c_dup1_1);
        struct nlop_s* T1c_link2 = nlop_link_F(T1c_combine2, 2, 0);
        struct nlop_s* T1c_dup2 = nlop_dup_F(T1c_link2, 2, 6);
        struct nlop_s* T1c_dup2_1 = nlop_dup_F(T1c_dup2, 1, 4);
        struct nlop_s* T1c_dup2_2 = nlop_dup_F(T1c_dup2_1, 0, 3);


        // third chain: T1(T1s(T1(T1s)))
        struct nlop_s* T1c_combine3 = nlop_combine_FF(T1_2, T1c_dup2_2);
        struct nlop_s* T1c_link3 = nlop_link_F(T1c_combine3, 2, 0);
        struct nlop_s* T1c_dup3 = nlop_dup_F(T1c_link3, 1, 3);
        struct nlop_s* T1c_dup3_1 = nlop_dup_F(T1c_dup3, 0, 2);

        //  stack the outputs together
        long sodims[N];
        md_copy_dims(N, sodims, out_dims);
        sodims[TE_DIM] = 2 * out_dims[TE_DIM];

        struct nlop_s* stack = nlop_stack_create(N, sodims, out_dims, out_dims, TE_DIM);
        struct nlop_s* T1c_combine_stack = nlop_combine_FF(stack, T1c_dup3_1);
        struct nlop_s* T1c_link_stack_1 = nlop_link_F(T1c_combine_stack, 3, 0);
        struct nlop_s* T1c_link_stack_2 = nlop_link_F(T1c_link_stack_1, 2, 0);

        // fourth chain : T1s(T1(T1s(T1(T1s))))
        struct nlop_s* T1c_combine4 = nlop_combine_FF(T1s_3, T1c_link_stack_2);
        struct nlop_s* T1c_link4 = nlop_link_F(T1c_combine4, 3, 0);
        struct nlop_s* T1c_dup4 = nlop_dup_F(T1c_link4, 2, 5);
        struct nlop_s* T1c_dup4_1 = nlop_dup_F(T1c_dup4, 1, 4);
        struct nlop_s* T1c_dup4_2 = nlop_dup_F(T1c_dup4_1, 0, 3);

        //  stack the outputs together
        long sodims1[N];
        md_copy_dims(N, sodims1, out_dims);
        sodims1[TE_DIM] = sodims[TE_DIM] + out_dims[TE_DIM];
        struct nlop_s* stack1 = nlop_stack_create(N, sodims1, sodims, out_dims, TE_DIM);
        struct nlop_s* T1c_combine_stack1 = nlop_combine_FF(stack1, T1c_dup4_2);
        struct nlop_s* T1c_link_stack_1_1 = nlop_link_F(T1c_combine_stack1, 3, 0);
        struct nlop_s* T1c_link_stack_2_1 = nlop_link_F(T1c_link_stack_1_1, 1, 0);

        
        // fifth chain : T1(T1s(T1(T1s(T1(T1s)))))
        struct nlop_s* T1c_combine5 = nlop_combine_FF(T1_3, T1c_link_stack_2_1);
        struct nlop_s* T1c_link5 = nlop_link_F(T1c_combine5, 2, 0);
        struct nlop_s* T1c_dup5 = nlop_dup_F(T1c_link5, 1, 3);
        struct nlop_s* T1c_dup5_1 = nlop_dup_F(T1c_dup5, 0, 2);

        // Sixth chain : T1s(T1(T1s(T1(T1s(T1(T1s))))))
        struct nlop_s* T1c_combine6 = nlop_combine_FF(T1s_4, T1c_dup5_1);
        struct nlop_s* T1c_link6 = nlop_link_F(T1c_combine6, 2, 0);
        struct nlop_s* T1c_dup6 = nlop_dup_F(T1c_link6, 2, 5);
        struct nlop_s* T1c_dup6_1 = nlop_dup_F(T1c_dup6, 1, 4);
        struct nlop_s* T1c_dup6_2 = nlop_dup_F(T1c_dup6_1, 0, 3);

         //  stack the outputs together
        long sodims2[N];
        md_copy_dims(N, sodims2, out_dims);
        sodims2[TE_DIM] = sodims1[TE_DIM] + out_dims[TE_DIM];
        struct nlop_s* stack2 = nlop_stack_create(N, sodims2, sodims1, out_dims, TE_DIM);
        struct nlop_s* T1c_combine_stack2 = nlop_combine_FF(stack2, T1c_dup6_2);
        struct nlop_s* T1c_link_stack_1_2 = nlop_link_F(T1c_combine_stack2, 3, 0);
        struct nlop_s* T1c_link_stack_2_2 = nlop_link_F(T1c_link_stack_1_2, 1, 0);

        // seventh chain : T1(T1s(T1(T1s(T1(T1s(T1(T1s)))))))
        struct nlop_s* T1c_combine7 = nlop_combine_FF(T1_4, T1c_link_stack_2_2);
        struct nlop_s* T1c_link7 = nlop_link_F(T1c_combine7, 2, 0);
        struct nlop_s* T1c_dup7 = nlop_dup_F(T1c_link7, 1, 3);
        struct nlop_s* T1c_dup7_1 = nlop_dup_F(T1c_dup7, 0, 2);

        // eighth chain : T1s(T1(T1s(T1(T1s(T1(T1s(T1(T1s))))))))
        struct nlop_s* T1c_combine8 = nlop_combine_FF(T1s_5, T1c_dup7_1);
        struct nlop_s* T1c_link8 = nlop_link_F(T1c_combine8, 2, 0);
        struct nlop_s* T1c_dup8 = nlop_dup_F(T1c_link8, 2, 5);
        struct nlop_s* T1c_dup8_1 = nlop_dup_F(T1c_dup8, 1, 4);
        struct nlop_s* T1c_dup8_2 = nlop_dup_F(T1c_dup8_1, 0, 3);


        //  stack the outputs together
        long sodims3[N];
        md_copy_dims(N, sodims3, out_dims);
        sodims3[TE_DIM] = sodims2[TE_DIM] + out_dims[TE_DIM];
        struct nlop_s* stack3 = nlop_stack_create(N, sodims3, sodims2, out_dims, TE_DIM);
        struct nlop_s* T1c_combine_stack3 = nlop_combine_FF(stack3, T1c_dup8_2);
        struct nlop_s* T1c_link_stack_1_3 = nlop_link_F(T1c_combine_stack3, 3, 0);
        struct nlop_s* T1c_link_stack_2_3 = nlop_link_F(T1c_link_stack_1_3, 1, 0);

        struct nlop_s* T1c_del = nlop_del_out(T1c_link_stack_2_3, 1);

        // // scaling operator
        complex float diag[1] = {-1.0};
        md_copy(N, scale_dims, scale, diag, CFL_SIZE);

        struct linop_s* linop_scalar = linop_cdiag_create(N, map_dims, COEFF_FLAG, scale);

        struct nlop_s* nl_scalar = nlop_from_linop(linop_scalar);

        linop_free(linop_scalar);

        struct nlop_s* T1c_combine_scale = nlop_combine_FF(T1c_del, nl_scalar);
        struct nlop_s* T1c_link_scale = nlop_link_F(T1c_combine_scale, 1, 3);
        struct nlop_s* T1c_dup_scale = nlop_dup_F(T1c_link_scale, 0, 3);


        md_free(scale);


        return T1c_dup_scale;
}