# Copyright 2014. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.


itersrcs := $(wildcard $(srcdir)/iter/*.c)
itercudasrcs := $(wildcard $(srcdir)/iter/*.cu)
iterobjs := $(itersrcs:.c=.o)

ifeq ($(CUDA),1)
iterobjs += $(itercudasrcs:.cu=.o)
endif


.INTERMEDIATE: $(iterobjs)

lib/libiter.a: libiter.a($(iterobjs))

UTARGETS += test_iter test_prox
MODULES_test_iter += -liter -lnlops -llinops -lwavelet
MODULES_test_prox += -liter -llinops

