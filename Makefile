# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2015-2019. Martin Uecker <martin.uecker@med.uni-goettingen.de>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.

# we have a two stage Makefile
MAKESTAGE ?= 1

# silent make
#MAKEFLAGS += --silent

# auto clean on makefile updates
AUTOCLEAN?=1

# clear out all implicit rules and variables
MAKEFLAGS += -R

# use for parallel make
AR=./ar_lock.sh

# allow blas calls within omp regions (fails on Debian 9, openblas)
BLAS_THREADSAFE?=

# some operations might still be non deterministic
NON_DETERMINISTIC?=0

OPENBLAS?=0

# use for ppc64le HPC
MKL?=0
CUDA?=0
CUDNN?=0
ACML?=0
UUID?=0
OMP?=1
SLINK?=0
DEBUG?=0
UBSAN?=0
FFTWTHREADS?=1
SCALAPACK?=0
ISMRMRD?=0
TENSORFLOW?=0
TF_VERSION?=1
NOEXEC_STACK?=0
PARALLEL?=0
PARALLEL_NJOBS?=

LOG_BACKEND?=0
LOG_SIEMENS_BACKEND?=0
LOG_ORCHESTRA_BACKEND?=0
LOG_GADGETRON_BACKEND?=0
ENABLE_MEM_CFL?=0
MEMONLY_CFL?=0


DESTDIR ?= /
PREFIX ?= usr/local/

BUILDTYPE = Linux
UNAME = $(shell uname -s)
NNAME = $(shell uname -n)

MYLINK=ln


ifeq ($(UNAME),Darwin)
	BUILDTYPE = MacOSX
	MYLINK = ln -s
endif

ifeq ($(BUILDTYPE), MacOSX)
	MACPORTS ?= 1
endif

ifeq ($(BUILDTYPE), Linux)
	# as the defaults changed on most Linux distributions
	# explicitly specify non-deterministic archives to not break make
	ARFLAGS ?= rsU
else
	ARFLAGS ?= rs
endif


ifeq ($(UNAME),Cygwin)
	BUILDTYPE = Cygwin
	NOLAPACKE ?= 1
endif

ifeq ($(UNAME),CYGWIN_NT-10.0)
	BUILDTYPE = Cygwin
	NOLAPACKE ?= 1
endif




# Paths

here  = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
root := $(here)

srcdir = $(root)/src
libdir = $(root)/lib
bindir = $(root)/bin

export TOOLBOX_PATH=$(root)


# Automatic dependency generation

DEPFILE = $(*D)/.$(*F).d
DEPFLAG = -MMD -MF $(DEPFILE)
ALLDEPS = $(shell find $(srcdir) utests -name ".*.d")


# Compilation flags

OPT = -O3 -ffast-math
CPPFLAGS ?= -Wall -Wextra
CFLAGS ?= $(OPT) -Wmissing-prototypes
CXXFLAGS ?= $(OPT)

ifeq ($(BUILDTYPE), MacOSX)
	CC ?= gcc-mp-6
else
	CC ?= gcc
	# for symbols in backtraces
	LDFLAGS += -rdynamic
endif




# openblas

ifneq ($(BUILDTYPE), MacOSX)
BLAS_BASE ?= /usr/
else
ifeq ($(MACPORTS),1)
BLAS_BASE ?= /opt/local/
CPPFLAGS += -DUSE_MACPORTS
endif
BLAS_BASE ?= /usr/local/opt/openblas/
endif

# cuda

CUDA_BASE ?= /usr/
CUDA_LIB ?= lib
CUDNN_BASE ?= $(CUDA_BASE)

# tensorflow
TENSORFLOW_BASE ?= /usr/local/

# acml

ACML_BASE ?= /usr/local/acml/acml4.4.0/gfortran64_mp/

# mkl
MKL_BASE ?= /opt/intel/mkl/

# fftw

ifneq ($(BUILDTYPE), MacOSX)
FFTW_BASE ?= /usr/
else
FFTW_BASE ?= /opt/local/
endif


# Matlab

MATLAB_BASE ?= /usr/local/matlab/

# ISMRM

ISMRM_BASE ?= /usr/local/ismrmrd/



# Main build targets
#
TBASE=show slice crop resize join transpose squeeze flatten zeros ones flip circshift extract repmat bitmask reshape version delta copy casorati vec poly index linspace pad morph multicfl fd
TFLP=scale invert conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg cabs zexp
TNUM=fft fftmod fftshift noise bench threshold conv rss filter mandelbrot wavelet window var std fftrot roistat pol2mask conway
TRECO=pics pocsense sqpics itsense nlinv T1fun moba mobafit cdi modbloch pixel nufft rof tgv sake wave lrmatrix estdims estshift estdelay wavepsf wshfl hornschunck ncsense kmat power approx kernel dcnn rtreco rtnlinv sudoku
TCALIB=ecalib ecaltwo caldir walsh cc ccapply calmat svd estvar whiten rmfreq ssa bin cordelay laplace kmeans convkern nlsa eof
TMRI=homodyne poisson twixread fakeksp umgread looklocker schmitt paradiseread phasediff dixon synthesize fovshift
TIO=toimg dcmread dcmtag
TSIM=phantom phantom_json traj upat bloch sim signal epg leray pde pde_mask bfield
TNN=mnist nnvn nnmodl reconet nnet onehotenc



MODULES = -lnum -lmisc -lnum -lmisc -lna

MODULES_pics = -lgrecon -lsense -liter -llinops -lwavelet -llowrank -lnoncart -lnlops -lnn -lmanifold
MODULES_sqpics = -lsense -liter -llinops -lwavelet -llowrank -lnoncart
MODULES_pocsense = -lsense -liter -llinops -lwavelet
MODULES_nlinv = -lnoir -liter -lnlops -llinops -lnoncart
MODULES_cdi = -liter -lsimu -llinops
MODULES_rtnlinv = -lnoir -liter -lnlops -llinops -lnoncart
MODULES_moba = -lmoba -lnoir -lnlops -llinops -lwavelet -lnoncart -lsimu -lgrecon -llowrank -llinops -liter -lnn
MODULES_mobafit = -lmoba -lnlops -llinops -lsimu -liter
MODULES_bpsense = -lsense -lnoncart -liter -llinops -lwavelet
MODULES_itsense = -liter -llinops
MODULES_ecalib = -lcalib
MODULES_ecaltwo = -lcalib
MODULES_estdelay = -lcalib
MODULES_caldir = -lcalib
MODULES_walsh = -lcalib
MODULES_calmat = -lcalib
MODULES_cc = -lcalib
MODULES_ccapply = -lcalib
MODULES_estvar = -lcalib
MODULES_nufft = -lnoncart -liter -llinops
MODULES_rof = -liter -llinops
MODULES_tgv = -liter -llinops
MODULES_bench = -lwavelet -llinops
MODULES_phantom = -lsimu -lgeom
MODULES_phantom_json = -lsimu -lgeom
MODULES_bart = -lbox -lgrecon -lsense -lnoir -liter -llinops -lwavelet -llowrank -lnoncart -lcalib -lsimu -lsake -ldfwavelet -lnlops -lrkhs -lnetworks -lnn -liter -lmanifold -lmoba -lgeom -lnlops
MODULES_sake = -lsake
MODULES_traj = -lnoncart
MODULES_wave = -liter -lwavelet -llinops -llowrank
MODULES_threshold = -llowrank -liter -ldfwavelet -llinops -lwavelet
MODULES_fakeksp = -lsense -llinops
MODULES_lrmatrix = -llowrank -liter -llinops -lnlops
MODULES_estdims = -lnoncart -llinops
MODULES_ismrmrd = -lismrm
MODULES_wavelet = -llinops -lwavelet
MODULES_wshfl = -lgrecon -lsense -liter -llinops -lwavelet -llowrank -lnoncart -lnlops -lnn
MODULES_hornschunck = -liter -llinops
MODULES_ncsense = -liter -llinops -lnoncart -lsense
MODULES_kernel = -lrkhs -lnoncart
MODULES_power = -lrkhs -lnoncart
MODULES_approx = -lrkhs -lnoncart
MODULES_kmat = -lrkhs -lnoncart
MODULES_dcnn = -lnetworks -lnn -lnlops -llinops -liter
MODULES_ssa = -lcalib -lmanifold -liter -llinops
MODULES_nlsa = -lcalib -lmanifold -liter -llinops
MODULES_bin = -lcalib
MODULES_laplace = -lmanifold -liter -llinops
MODULES_kmeans = -lmanifold -liter -llinops
MODULES_tgv = -liter -llinops
MODULES_bloch = -lsimu -lgeom
MODULES_modbloch = -lmoba -lnoir -liter -lsimu -lnlops -lwavelet -lnoncart -lgrecon -llinops -llowrank -lnn
MODULES_sim = -lsimu
MODULES_rtnlinv = -lnoncart -lnoir -lnlops -liter -llinops
MODULES_signal = -lsimu
MODULES_eof = -lcalib -lmanifold
MODULES_pol2mask = -lgeom
MODULES_sudoku = -llinops -liter -lnlops
MODULES_nnvn = -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_nnmodl = -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_reconet = -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_nnet = -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_onehotenc = -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_mnist = -lnetworks -lnn -lnlops -llinops -liter
MODULES_morph = -lnlops -llinops -lgeom
MODULES_epg = -lsimu
MODULES_fd = -llinops
MODULES_leray = -lsimu -llinops -liter
MODULES_pde = -lsimu -liter -llinops
MODULES_pde_mask = -lsimu -llinops
MODULES_bfield = -lsimu -llinops
MODULES_dixon = -lmoba -lnlops -llinops -lsimu
MODULES_pixel = -lmoba -lnoir -liter -lsimu -lnlops -lwavelet -lgrecon -lnoncart -llinops -llowrank -lnn
MODULES_rtreco = -lcalib -lnoncart -llinops

MAKEFILES = $(wildcard $(root)/Makefiles/Makefile.*)
ALLMAKEFILES = $(root)/Makefile $(wildcard $(root)/Makefile.* $(root)/*.mk $(root)/rules/*.mk $(root)/Makefiles/Makefile.*)

-include Makefile.$(NNAME)
-include Makefile.local
-include $(MAKEFILES)


# clang

ifeq ($(findstring clang,$(CC)),clang)
	CFLAGS += -fblocks
	LDFLAGS += -lBlocksRuntime
endif


CXX ?= g++
LINKER ?= $(CC)



ifeq ($(ISMRMRD),1)
TMRI += ismrmrd
MODULES_bart += -lismrm
endif

ifeq ($(NOLAPACKE),1)
CPPFLAGS += -DNOLAPACKE
MODULES += -llapacke
endif

ifeq ($(TENSORFLOW),1)
CPPFLAGS += -DTENSORFLOW -DTF_VERSION -I$(TENSORFLOW_BASE)/include
LIBS += -L$(TENSORFLOW_BASE)/lib -Wl,-rpath $(TENSORFLOW_BASE)/lib -ltensorflow -ltensorflow_framework
endif


XTARGETS += $(TBASE) $(TFLP) $(TNUM) $(TIO) $(TRECO) $(TCALIB) $(TMRI) $(TSIM) $(TNN)
TARGETS = bart $(XTARGETS)



ifeq ($(DEBUG),1)
CPPFLAGS += -g
CFLAGS += -g
NVCCFLAGS += -g
endif

ifeq ($(UBSAN),1)
CFLAGS += -fsanitize=undefined -fsanitize-undefined-trap-on-error
endif

ifeq ($(NOEXEC_STACK),1)
CPPFLAGS += -DNOEXEC_STACK
endif


ifeq ($(PARALLEL),1)
MAKEFLAGS += -j$(PARALLEL_NJOBS)
endif


ifeq ($(MAKESTAGE),1)
.PHONY: doc/commands.txt $(TARGETS)
default all clean allclean distclean doc/commands.txt doxygen test utest utest_gpu gputest pythontest testague testslow $(TARGETS):
	$(MAKE) MAKESTAGE=2 $(MAKECMDGOALS)

tests/test-%: force
	$(MAKE) MAKESTAGE=2 $(MAKECMDGOALS)

force: ;

else


CPPFLAGS += $(DEPFLAG) -iquote $(srcdir)/
CFLAGS += -std=gnu11
CXXFLAGS += -std=c++14




default: bart doc/commands.txt .gitignore


-include $(ALLDEPS)




# cuda

NVCC = $(CUDA_BASE)/bin/nvcc


ifeq ($(CUDA),1)
CUDA_H := -I$(CUDA_BASE)/include
CPPFLAGS += -DUSE_CUDA $(CUDA_H)
ifeq ($(CUDNN),1)
CUDNN_H := -I$(CUDNN_BASE)/include
CPPFLAGS += -DUSE_CUDNN $(CUDNN_H)
endif
ifeq ($(BUILDTYPE), MacOSX)
CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -lcufft -lcudart -lcublas -m64 -lstdc++
else
ifeq ($(CUDNN),1)
CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -L$(CUDNN_BASE)/lib64 -lcudnn -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/$(CUDA_LIB)
else
CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/$(CUDA_LIB)
endif
endif
else
CUDA_H :=
CUDA_L :=
endif

# sm_20 no longer supported in CUDA 9
GPUARCH_FLAGS ?= -arch=compute_50
NVCCFLAGS += -DUSE_CUDA -Xcompiler -fPIC -Xcompiler -fopenmp -O3 $(GPUARCH_FLAGS) -I$(srcdir)/ -m64 -ccbin $(CC)
#NVCCFLAGS = -Xcompiler -fPIC -Xcompiler -fopenmp -O3  -I$(srcdir)/


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@
	$(NVCC) $(NVCCFLAGS) -M $^ -o $(DEPFILE)


# OpenMP

ifeq ($(OMP),1)
CFLAGS += -fopenmp
CXXFLAGS += -fopenmp
else
CFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unknown-pragmas
endif

#CFLAGS += -DSSAFARY_PAPER # For reproduction of SSA-FARY paper!


# BLAS/LAPACK
ifeq ($(SCALAPACK),1)
BLAS_L :=  -lopenblas -lscalapack
else
ifeq ($(ACML),1)
BLAS_H := -I$(ACML_BASE)/include
BLAS_L := -L$(ACML_BASE)/lib -lgfortran -lacml_mp -Wl,-rpath $(ACML_BASE)/lib
CPPFLAGS += -DUSE_ACML
else
BLAS_H := -I$(BLAS_BASE)/include
ifeq ($(BUILDTYPE), MacOSX)
BLAS_L := -L$(BLAS_BASE)/lib -lopenblas
else
ifeq ($(NOLAPACKE),1)
BLAS_L := -L$(BLAS_BASE)/lib -llapack -lblas
CPPFLAGS += -Isrc/lapacke
else
ifeq ($(OPENBLAS), 1)
BLAS_L := -L$(BLAS_BASE)/lib -llapacke -lopenblas
CPPFLAGS += -DUSE_OPENBLAS
CFLAGS += -DUSE_OPENBLAS
BLAS_THREADSAFE?=1
else
BLAS_L := -L$(BLAS_BASE)/lib -llapacke -lblas
endif
endif
endif
endif
endif

ifeq ($(MKL),1)
BLAS_H := -I$(MKL_BASE)/include
BLAS_L := -L$(MKL_BASE)/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
CPPFLAGS += -DUSE_MKL -DMKL_Complex8="complex float" -DMKL_Complex16="complex double"
CFLAGS += -DUSE_MKL -DMKL_Complex8="complex float" -DMKL_Complex16="complex double"
BLAS_THREADSAFE?=1
endif

BLAS_THREADSAFE?=0
ifeq ($(BLAS_THREADSAFE),1)
CPPFLAGS += -DBLAS_THREADSAFE
CFLAGS += -DBLAS_THREADSAFE
endif

ifeq ($(NON_DETERMINISTIC),1)
CPPFLAGS += -DNON_DETERMINISTIC
CFLAGS += -DNON_DETERMINISTIC
NVCCFLAGS += -DNON_DETERMINISTIC
endif


CPPFLAGS += $(FFTW_H) $(BLAS_H)



# png
PNG_L := -lpng

ifeq ($(SLINK),1)
	PNG_L += -lz
endif

ifeq ($(LINKER),icc)
	PNG_L += -lz
endif


# uuid
ifeq ($(UUID), 1)
	LDFLAGS += -luuid
	CPPFLAGS += -DHAVE_UUID
endif


# fftw

FFTW_H := -I$(FFTW_BASE)/include/
FFTW_L := -L$(FFTW_BASE)/lib -lfftw3f

ifeq ($(FFTWTHREADS),1)
	FFTW_L += -lfftw3f_threads
	CPPFLAGS += -DFFTWTHREADS
endif

# Matlab

MATLAB_H := -I$(MATLAB_BASE)/extern/include
MATLAB_L := -Wl,-rpath $(MATLAB_BASE)/bin/glnxa64 -L$(MATLAB_BASE)/bin/glnxa64 -lmat -lmx -lm -lstdc++

# ISMRM

ifeq ($(ISMRMRD),1)
ISMRM_H := -I$(ISMRM_BASE)/include
ISMRM_L := -L$(ISMRM_BASE)/lib -lismrmrd
else
ISMRM_H :=
ISMRM_L :=
endif

# Enable in-memory CFL files

ifeq ($(ENABLE_MEM_CFL),1)
CPPFLAGS += -DUSE_MEM_CFL
miscextracxxsrcs += $(srcdir)/misc/mmiocc.cc
LDFLAGS += -lstdc++
endif

# Only allow in-memory CFL files (ie. disable support for all other files)

ifeq ($(MEMONLY_CFL),1)
CPPFLAGS += -DMEMONLY_CFL
miscextracxxsrcs += $(srcdir)/misc/mmiocc.cc
LDFLAGS += -lstdc++
endif

# Logging backends

ifeq ($(LOG_BACKEND),1)
CPPFLAGS += -DUSE_LOG_BACKEND
ifeq ($(LOG_SIEMENS_BACKEND),1)
miscextracxxsrcs += $(srcdir)/misc/UTrace.cc
endif
ifeq ($(LOG_ORCHESTRA_BACKEND),1)
miscextracxxsrcs += $(srcdir)/misc/Orchestra.cc
endif
endif


# change for static linking

ifeq ($(SLINK),1)
ifeq ($(SCALAPACK),1)
BLAS_L += -lgfortran -lquadmath
else
# work around fortran problems with static linking
LDFLAGS += -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition
LIBS += -lmvec
BLAS_L += -llapack -lblas -lgfortran -lquadmath
endif
endif



# Modules

.LIBPATTERNS := lib%.a

vpath %.a lib

boxextrasrcs := $(XTARGETS:%=src/%.c)

define alib
$(1)srcs := $(wildcard $(srcdir)/$(1)/*.c)
$(1)cudasrcs := $(wildcard $(srcdir)/$(1)/*.cu)
$(1)objs := $$($(1)srcs:.c=.o)
$(1)objs += $$($(1)extrasrcs:.c=.o)
$(1)objs += $$($(1)extracxxsrcs:.cc=.o)

ifeq ($(CUDA),1)
$(1)objs += $$($(1)cudasrcs:.cu=.o)
endif

.INTERMEDIATE: $$($(1)objs)

lib/lib$(1).a: lib$(1).a($$($(1)objs))

endef

ALIBS = misc num grecon sense noir iter linops wavelet lowrank noncart calib simu sake dfwavelet nlops moba lapacke box geom rkhs na networks nn manifold seq
ifeq ($(ISMRMRD),1)
ALIBS += ismrm
endif

$(eval $(foreach t,$(ALIBS),$(eval $(call alib,$(t)))))


# additional rules for lib misc
$(eval $(shell $(root)/rules/update-version.sh))

$(srcdir)/misc/version.o: $(srcdir)/misc/version.inc


# additional rules for lib ismrm
lib/libismrm.a: CPPFLAGS += $(ISMRM_H)


# lib linop
UTARGETS += test_linop_matrix test_linop test_linop_conv test_linop_fd
MODULES_test_linop += -llinops
MODULES_test_linop_matrix += -llinops
MODULES_test_linop_conv += -llinops
MODULES_test_linop_fd += -llinops

# lib lowrank
UTARGETS += test_batchsvd
MODULES_test_batchsvd = -llowrank

# lib misc
UTARGETS += test_pattern test_types test_misc test_mmio

# lib moba
UTARGETS += test_moba test_scale test_bloch_op
MODULES_test_moba += -lmoba -lnoir -llowrank -lwavelet -liter -lnlops -llinops -lsimu
MODULES_test_scale += -lmoba -lnoir -llowrank -liter -lnlops -llinops -lsimu
MODULES_test_bloch_op += -lmoba -lnoir -llowrank -lwavelet -liter -lnlops -llinops -lsimu

# lib nlop
UTARGETS += test_nlop
MODULES_test_nlop += -lnlops -llinops  -liter

# lib noncart
UTARGETS += test_nufft
MODULES_test_nufft += -lnoncart -llinops

# lib num
UTARGETS += test_multind test_flpmath test_splines test_linalg test_polynom test_window test_mat2x2 test_fdiff
UTARGETS += test_blas test_mdfft test_filter test_conv test_ops test_matexp test_ops_p test_specfun test_convcorr test_flpmath2
UTARGETS_GPU += test_cuda_gpukrnls test_cudafft test_cuda_flpmath2 test_cuda_memcache_clear test_cuda_flpmath

# lib simu
UTARGETS += test_ode_bloch test_tsegf test_biot_savart test_biot_savart_fft test_ode_simu test_ode_pulse test_signals test_epg test_crb test_fd_geometry test_sparse test_pde test_linop_leray
MODULES_test_ode_bloch += -lsimu
MODULES_test_tsegf += -lsimu
MODULES_test_biot_savart += -lsimu
MODULES_test_biot_savart_fft += -lsimu -llinops
MODULES_test_ode_simu += -lsimu
MODULES_test_ode_pulse += -lsimu
MODULES_test_signals += -lsimu
MODULES_test_epg += -lsimu
MODULES_test_crb += -lsimu
MODULES_test_linop_leray += -lsimu -liter -llinops
MODULES_test_fd_geometry += -lsimu -llinops
MODULES_test_sparse += -lsimu -llinops
MODULES_test_pde += -lsimu -llinops

# lib slice profile
UTARGETS +=test_slice_profile
MODULES_test_slice_profile += -lsimu

# lib geom
UTARGETS += test_geom
MODULES_test_geom += -lgeom

# lib iter
UTARGETS += test_iter test_prox test_prox2
MODULES_test_iter += -liter -lnlops -llinops
MODULES_test_prox += -liter -llinops
MODULES_test_prox2 += -liter -llinops -lnlops

# lib nn
UTARGETS += test_nn_ops test_nn
MODULES_test_nn_ops += -lnn -lnlops -llinops -liter
MODULES_test_nn += -lnn -lnlops -llinops -liter

UTARGETS_GPU += test_cuda_nlop
MODULES_test_cuda_nlop += -lnn -lnlops -llinops -lnum

# lib calib
UTARGETS += test_eof
MODULES_test_eof += -lcalib -lnum -lmanifold -liter -llinops

# sort BTARGETS after everything is included
BTARGETS:=$(sort $(BTARGETS))
XTARGETS:=$(sort $(XTARGETS))



.gitignore: .gitignore.main Makefile*
	@echo '# AUTOGENERATED. DO NOT EDIT. (are you looking for .gitignore.main ?)' > .gitignore
	cat .gitignore.main >> .gitignore
	@echo $(patsubst %, /%, $(TARGETS) $(UTARGETS) $(UTARGETS_GPU)) | tr ' ' '\n' >> .gitignore


doc/commands.txt: bart
	./rules/update_commands.sh ./bart doc/commands.txt $(XTARGETS)

doxygen: makedoc.sh doxyconfig bart
	 ./makedoc.sh


all: .gitignore $(TARGETS)





# special targets


$(XTARGETS): CPPFLAGS += -DMAIN_LIST="$(XTARGETS:%=%,) ()" -include src/main.h


bart: CPPFLAGS += -DMAIN_LIST="$(XTARGETS:%=%,) ()" -include src/main.h


mat2cfl: $(srcdir)/mat2cfl.c -lnum -lmisc
	$(CC) $(CFLAGS) $(MATLAB_H) -omat2cfl  $+ $(MATLAB_L) $(CUDA_L)





# implicit rules

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

ifeq ($(PARALLEL),1)
(%): %
	$(AR) $(ARFLAGS) $@ $%
else
(%): %
	$(AR) $(ARFLAGS) $@ $%
endif




.SECONDEXPANSION:
$(TARGETS): % : src/main.c $(srcdir)/%.o $$(MODULES_%) $(MODULES)
	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$@ -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm -lrt
#	rm $(srcdir)/$@.o

UTESTS=$(shell $(root)/utests/utests-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS): % : utests/utest.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS="$(UTESTS)" -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm -lrt

UTESTS_GPU=$(shell $(root)/utests/utests_gpu-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS_GPU): % : utests/utest_gpu.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS_GPU="$(UTESTS_GPU)" -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm -lrt



# linker script version - does not work on MacOS X
#	$(CC) $(LDFLAGS) -Wl,-Tutests/utests.ld $(CFLAGS) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) -lm -rt

clean:
	rm -f `find $(srcdir) -name "*.o"`
	rm -f $(root)/utests/*.o
	rm -f $(patsubst %, %, $(UTARGETS))
	rm -f $(libdir)/.*.lock

allclean: clean
	rm -f $(libdir)/*.a $(ALLDEPS)
	rm -f $(patsubst %, %, $(TARGETS))
	rm -f $(srcdir)/misc/version.inc
	rm -rf $(root)/tests/tmp/*/
	rm -rf $(root)/doc/dx
	rm -f $(root)/doc/commands.txt
	touch isclean

distclean: allclean



-include isclean


isclean: $(ALLMAKEFILES)
ifeq ($(AUTOCLEAN),1)
	@echo "CONFIGURATION MODIFIED. RUNNING FULL REBUILD."
	touch isclean
	$(MAKE) allclean || rm isclean
else
ifneq ($(MAKECMDGOALS),allclean)
	@echo "CONFIGURATION MODIFIED."
endif
endif



# automatic tests

# system tests

TOOLDIR=$(root)
TESTS_TMP=$(root)/tests/tmp/$$$$/
TESTS_OUT=$(root)/tests/out/


include $(root)/tests/*.mk

test:	${TESTS}
	@echo -n "The following tools do not have tests: "
	for i in ${XTARGETS} ; do \
		if [ ! -f ${root}/tests/$$i.mk ] ; then \
			echo -n "$$i "; \
		fi ; \
	done
	@echo

testslow: ${TESTS_SLOW}

testague: ${TESTS_AGUE} # test importing *.dat-files specified in tests/twixread.mk

gputest: ${TESTS_GPU}

pythontest: ${TESTS_PYTHON}

# unit tests

UTEST_RUN=

ifeq ($(UTESTLEAK),1)
# we blacklist some targets because valgrind crashes (blas related)
UTARGETS:=$(filter-out test_flpmath test_blas,$(UTARGETS))
UTEST_RUN=valgrind --quiet --leak-check=full --error-exitcode=1 valgrind --suppressions=./valgrind.supp --log-file=/dev/null
endif

# define space to faciliate running executables
define \n


endef

utests-all: $(UTARGETS)
	$(patsubst %,$(\n)$(UTEST_RUN) ./%,$(UTARGETS))

utest: utests-all
	@echo ALL CPU UNIT TESTS PASSED.

utests_gpu-all: $(UTARGETS_GPU)
	$(patsubst %,$(\n)$(UTEST_RUN) ./%,$(UTARGETS_GPU))

utest_gpu: utests_gpu-all
	@echo ALL GPU UNIT TESTS PASSED.



endif	# MAKESTAGE


install: bart $(root)/doc/commands.txt
	install -d $(DESTDIR)/$(PREFIX)/bin/
	install bart $(DESTDIR)/$(PREFIX)/bin/
	install -d $(DESTDIR)/$(PREFIX)/share/doc/bart/
	install $(root)/doc/*.txt $(root)/README $(DESTDIR)/$(PREFIX)/share/doc/bart/
	install -d $(DESTDIR)/$(PREFIX)/lib/bart/commands/


# generate release tar balls (identical to github)
%.tar.gz:
	git archive --prefix=bart-$(patsubst bart-%.tar.gz,%,$@)/ -o $@ v$(patsubst bart-%.tar.gz,%,$@)



# symbol table
bart.syms: bart
	rules/make_symbol_table.sh bart bart.syms
