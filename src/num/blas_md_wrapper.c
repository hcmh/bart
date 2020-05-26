/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "num/blas.h"

#include "blas_md_wrapper.h"
#include "num/multind.h"



/****************************************************************************************************
 *
 * Wrappers for zfmac
 *
 ****************************************************************************************************/

/**
 * cgemm for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 0, dim[0] * 8}
 * @param optr
 * @param istr1 must be of the form {8, dim[0] * 8, 0} or {dim[1] * 8, 8, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 8, dim[1] * 8} or {0, 8 * dim[2], 8}
 * @param iptr1 
 **/
void blas_zfmac_cgemm(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;

        assert(size == ostr[0]);
        assert(0 == ostr[1]);
        assert(0 == ostr[2] % size);
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1] % size);
        assert(0 == istr1[2]);
        assert(0 == istr2[0]);
        assert(0 == istr2[1] % size);
        assert(0 == istr2[2] % size);

    	char transa = (size == istr1[0]) ? 'N' : 'T';
        char transb = (size == istr2[1]) ? 'N' : 'T';

        long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
        long ldb = (size == istr2[1]) ? istr2[2] / size : istr2[1] / size;
        long ldc = ostr[2] / size;
        
        lda = MAX(1, lda);
        ldb = MAX(1, ldb);
        ldc = MAX(1, ldc);

	blas_cgemm(transa, transb, dims[0], dims[2], dims[1], 1., lda, iptr1, ldb, iptr2, 1., ldc, optr);
}

/**
 * cgemv for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 0}
 * @param optr
 * @param istr1 must be of the form {8, dim[0] * 8} or {dim[1] * 8, 8}
 * @param iptr1
 * @param istr1 must be of the form {0, 8}
 * @param iptr1 
 **/
void blas_zfmac_cgemv(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;

        assert(0 == ostr[0] % size);
        assert(0 == ostr[1]);
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1] % size);
        assert(0 == istr2[0]);
        assert(0 == istr2[1] % size);

    	char trans = (size == istr1[0]) ? 'N' : 'T';

        long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
        long incx = istr2[1] / size;
        long incy = ostr[0] / size; 
        
        lda = MAX(1, lda);

        blas_cgemv(trans, dims[0], dims[1], 1., lda, iptr1, incx, iptr2, 1., incy, optr);
}

/**
 * cgemv for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dims[0]}
 * @param optr
 * @param istr1 must be of the form {8, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 8}
 * @param iptr1 
 **/
void blas_zfmac_cgeru(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;

        assert(2 == N);
        assert(size == ostr[0]);
        assert((0 == ostr[1] % size) && (size * dims[0] <= ostr[1]));
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1]);
        assert(0 == istr2[0]);
        assert(0 == istr2[1] % size);

        long lda = ostr[1] / size;
        long incx = istr1[0] / size;
        long incy = istr2[1] / size; 
        
        lda = MAX(1, lda);

        blas_cgeru(dims[0], dims[1], 1., incx, iptr1, incy, iptr2, lda, optr);
}

/**
 * cgemv for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8}
 * @param optr
 * @param istr1 must be of the form {8}
 * @param iptr1
 * @param istr2 must be of the form {0}
 * @param iptr2 
 **/
void blas_zfmac_caxpy(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;

        assert(1 == N);
        assert(0 == ostr[0] % size);
        assert(0 == istr1[0] % size);
        assert(0 == istr2[0]);

        long incx = istr1[0] / size;
        long incy = ostr[0] / size; 

        blas2_caxpy(dims[0], iptr2, incx, iptr1, incy, optr);
}

/**
 * cdotu for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {0}
 * @param optr
 * @param istr1 must be of the form {8}
 * @param iptr1
 * @param istr2 must be of the form {8}
 * @param iptr2 
 **/
void blas_zfmac_cdotu(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;

        assert(1 == N);
        assert(0 == ostr[0]);
        assert(0 == istr1[0] % size);
        assert(0 == istr2[0] % size);

        long incx = istr1[0] / size;
        long incy = istr2[0] / size; 

        complex float* tmp = md_alloc_sameplace(1, MAKE_ARRAY(1l), size, optr);
        blas2_cdotu(tmp, dims[0], incx, iptr1, incy, iptr2);
        blas_caxpy(1, 1., size, tmp, size, optr);
        md_free(tmp);
}

/****************************************************************************************************
 *
 * Wrappers for zfmac
 *
 ****************************************************************************************************/

/**
 * cgemm for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {4, 0, dim[0] * 4}
 * @param optr
 * @param istr1 must be of the form {4, dim[0] * 4, 0} or {dim[1] * 4, 4, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 4, dim[1] * 4} or {0, 4 * dim[2], 4}
 * @param iptr1 
 **/
void blas_fmac_sgemm(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;

        assert(size == ostr[0]);
        assert(0 == ostr[1]);
        assert(0 == ostr[2] % size);
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1] % size);
        assert(0 == istr1[2]);
        assert(0 == istr2[0]);
        assert(0 == istr2[1] % size);
        assert(0 == istr2[2] % size);

    	char transa = (size == istr1[0]) ? 'N' : 'T';
        char transb = (size == istr2[1]) ? 'N' : 'T';

        long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
        long ldb = (size == istr2[1]) ? istr2[2] / size : istr2[1] / size;
        long ldc = ostr[2] / size;
        
        lda = MAX(1, lda);
        ldb = MAX(1, ldb);
        ldc = MAX(1, ldc);

	blas_sgemm(transa, transb, dims[0], dims[2], dims[1], 1., lda, iptr1, ldb, iptr2, 1., ldc, optr);
}

/**
 * sgemv for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 0}
 * @param optr
 * @param istr1 must be of the form {8, dim[0] * 8} or {dim[1] * 8, 8}
 * @param iptr1
 * @param istr1 must be of the form {0, 8}
 * @param iptr1 
 **/
void blas_fmac_sgemv(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;

        assert(0 == ostr[0] % size);
        assert(0 == ostr[1]);
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1] % size);
        assert(0 == istr2[0]);
        assert(0 == istr2[1] % size);

    	char trans = (size == istr1[0]) ? 'N' : 'T';

        long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
        long incx = istr2[1] / size;
        long incy = ostr[0] / size; 
        
        lda = MAX(1, lda);

        blas_sgemv(trans, dims[0], dims[1], 1., lda, iptr1, incx, iptr2, 1., incy, optr);
}

/**
 * cgemv for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dims[0]}
 * @param optr
 * @param istr1 must be of the form {8, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 8}
 * @param iptr1 
 **/
void blas_fmac_sger(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;

        assert(2 == N);
        assert(size == ostr[0]);
        assert((0 ==ostr[1] % size) && (size * dims[0] <= ostr[1]));
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1]);
        assert(0 == istr2[0]);
        assert(0 == istr2[1] % size);

        long lda = ostr[1] / size;
        long incx = istr1[0] / size;
        long incy = istr2[1] / size; 
        
        lda = MAX(1, lda);

        blas_sger(dims[0], dims[1], 1., incx, iptr1, incy, iptr2, lda, optr);
}

/**
 * cgemv for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {4}
 * @param optr
 * @param istr1 must be of the form {4}
 * @param iptr1
 * @param istr2 must be of the form {4}
 * @param iptr2 
 **/
void blas_fmac_saxpy(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;

        assert(1 == N);
        assert(0 == ostr[0] % size);
        assert(0 == istr1[0] % size);
        assert(0 == istr2[0]);

        long incx = istr1[0] / size;
        long incy = ostr[0] / size; 

        blas2_saxpy(dims[0], iptr2, incx, iptr1, incy, optr);
}

/**
 * cdotu for inner zfmac kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {0}
 * @param optr
 * @param istr1 must be of the form {8}
 * @param iptr1
 * @param istr2 must be of the form {8}
 * @param iptr2 
 **/
void blas_fmac_sdot(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;

        assert(1 == N);
        assert(0 == ostr[0]);
        assert(0 == istr1[0] % size);
        assert(0 == istr2[0] % size);

        long incx = istr1[0] / size;
        long incy = istr2[0] / size; 

        float* tmp = md_alloc_sameplace(1, MAKE_ARRAY(1l), size, optr);
        blas2_sdot(tmp, dims[0], incx, iptr1, incy, iptr2);
        blas_saxpy(1, 1., size, tmp, size, optr);
        md_free(tmp);
}

/****************************************************************************************************
 *
 * Wrappers for zmul / zsmul
 *
 ****************************************************************************************************/

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dim[0]}
 * @param optr
 * @param istr1 must be of the form {8, 8 * dim[0]} or {8 * dim[1], 8}
 * @param iptr1
 * @param istr1 must be of the form {0, 0}
 * @param iptr1 
 **/
void blas_zmul_cmatcopy(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;

        assert(size == ostr[0]);
        assert((0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1] % size);
        assert(0 == istr2[0]);
        assert(0 == istr2[1]);

    	char trans = (size == istr1[0]) ? 'N' : 'T';

        long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
        long ldb = ostr[1] / size;
        
        lda = MAX(1, lda);
        ldb = MAX(1, ldb);

	blas2_cmatcopy(trans, dims[0], dims[1], iptr2, iptr1, lda, optr, ldb);
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dim[0]}
 * @param optr
 * @param istr must be of the form {8, 8 * dim[0]} or {8 * dim[1], 8}
 * @param iptr
 * @param val
 **/
void blas_zsmul_cmatcopy(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr[N], const complex float* iptr, complex float val)
{
        long size = 8;

        assert(size == ostr[0]);
        assert((0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
        assert(0 == istr[0] % size);
        assert(0 == istr[1] % size);

    	char trans = (size == istr[0]) ? 'N' : 'T';

        long lda = (size == istr[0]) ? istr[1] / size : istr[0] / size;
        long ldb = ostr[1] / size;
        
        lda = MAX(1, lda);
        ldb = MAX(1, ldb);

	blas_cmatcopy(trans, dims[0], dims[1], val, iptr, lda, optr, ldb);
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dim[0]}
 * @param optr
 * @param istr1 must be of the form {8, 8 * dim[0]}
 * @param iptr1
 * @param istr1 must be of the form {8, 0} or {0, 8}
 * @param iptr1 
 **/
void blas_zmul_cdgmm(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;

        assert(size == ostr[0]);
        assert((0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
        assert(size == istr1[0]);
        assert((0 == istr1[1] % size) && (dims[0] * istr1[0] <= istr1[1]));
        assert((0 == istr2[0]) || (0 == istr2[1]));
        assert(0 == istr2[0] % size);
        assert(0 == istr2[1] % size);

        long lda = istr1[1] / size;
        long ldc = ostr[1] / size;
        long incx = (0 == istr2[1]) ? istr2[0] / size : istr2[1] / size;
        lda = MAX(1, lda);
        ldc = MAX(1, ldc);

        blas_cdgmm(dims[0], dims[1], 0 == istr2[1], iptr1, lda, iptr2, incx, optr, ldc);
}

/**
 * sger for inner mul kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dims[0]}
 * @param optr
 * @param istr1 must be of the form {8, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 8}
 * @param iptr1 
 **/
void blas_zmul_cgeru(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;
        md_clear2(N, dims, ostr, optr, size);
        blas_zfmac_cgeru(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}


/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8}
 * @param optr
 * @param istr1 must be of the form {8}
 * @param iptr1
 * @param istr1 must be of the form {0}
 * @param iptr1 
 **/
void blas_zmul_cscal(unsigned int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
{
        long size = 8;
        if (optr != iptr1) {

                md_clear2(N, dims, ostr, optr, size);
                blas_zfmac_caxpy(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
        } else {
                assert((ostr[0] == istr1[0]) && (0 == ostr[0] % size) && (0 == istr2[0]));
                assert(1 == N);
                blas2_cscal(dims[0], iptr2, istr1[0] / size, optr);
        }
        
}


/****************************************************************************************************
 *
 * Wrappers for mul / smul
 *
 ****************************************************************************************************/

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dim[0]}
 * @param optr
 * @param istr1 must be of the form {8, 8 * dim[0]} or {8 * dim[1], 8}
 * @param iptr1
 * @param istr1 must be of the form {0, 0}
 * @param iptr1 
 **/
void blas_mul_smatcopy(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;

        assert(size == ostr[0]);
        assert((0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
        assert(0 == istr1[0] % size);
        assert(0 == istr1[1] % size);
        assert(0 == istr2[0]);
        assert(0 == istr2[1]);

    	char trans = (size == istr1[0]) ? 'N' : 'T';

        long lda = (size == istr1[0]) ? istr1[1] / size : istr1[0] / size;
        long ldb = ostr[1] / size;
        
        lda = MAX(1, lda);
        ldb = MAX(1, ldb);

	blas2_smatcopy(trans, dims[0], dims[1], iptr2, iptr1, lda, optr, ldb);
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dim[0]}
 * @param optr
 * @param istr must be of the form {8, 8 * dim[0]} or {8 * dim[1], 8}
 * @param iptr
 * @param val
 **/
void blas_smul_smatcopy(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr[N], const float* iptr, float val)
{
        long size = 4;

        assert(size == ostr[0]);
        assert((0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
        assert(0 == istr[0] % size);
        assert(0 == istr[1] % size);

    	char trans = (size == istr[0]) ? 'N' : 'T';

        long lda = (size == istr[0]) ? istr[1] / size : istr[0] / size;
        long ldb = ostr[1] / size;
        
        lda = MAX(1, lda);
        ldb = MAX(1, ldb);

	blas_smatcopy(trans, dims[0], dims[1], val, iptr, lda, optr, ldb);
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dim[0]}
 * @param optr
 * @param istr1 must be of the form {8, 8 * dim[0]}
 * @param iptr1
 * @param istr1 must be of the form {8, 0} or {0, 8}
 * @param iptr1 
 **/
void blas_mul_sdgmm(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;

        assert(size == ostr[0]);
        assert((0 == ostr[1] % size) && (dims[0] * ostr[0] <= ostr[1]));
        assert(size == istr1[0]);
        assert((0 == istr1[1] % size) && (dims[0] * istr1[0] <= istr1[1]));
        assert((0 == istr2[0]) || (0 == istr2[1]));
        assert(0 == istr2[0] % size);
        assert(0 == istr2[1] % size);

        long lda = istr1[1] / size;
        long ldc = ostr[1] / size;
        long incx = (0 == istr2[1]) ? istr2[0] / size : istr2[1] / size;
        lda = MAX(1, lda);
        ldc = MAX(1, ldc);

        blas_sdgmm(dims[0], dims[1], 0 == istr2[1], iptr1, lda, iptr2, incx, optr, ldc);
}

/**
 * sger for inner mul kernel
 *
 * @param dims dimension
 * @param ostr must be of the form {8, 8 * dims[0]}
 * @param optr
 * @param istr1 must be of the form {8, 0}
 * @param iptr1
 * @param istr1 must be of the form {0, 8}
 * @param iptr1 
 **/
void blas_mul_sger(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;
        md_clear2(N, dims, ostr, optr, size);
        blas_fmac_sger(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
}

/**
 *
 * @param dims dimension
 * @param ostr must be of the form {8}
 * @param optr
 * @param istr1 must be of the form {8}
 * @param iptr1
 * @param istr1 must be of the form {0}
 * @param iptr1 
 **/
void blas_mul_sscal(unsigned int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
{
        long size = 4;
        if (optr != iptr1) {

                md_clear2(N, dims, ostr, optr, size);
                blas_fmac_saxpy(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
        } else {
                assert((ostr[0] == istr1[0]) && (0 == ostr[0] % size) && (0 == istr2[0]));
                assert(1 == N);
                blas2_sscal(dims[0], iptr2, istr1[0] / size, optr);
        }
        
}