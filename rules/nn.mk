# Copyright 2017. Martin Uecker.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.


nnsrcs := $(wildcard $(srcdir)/nn/*.c)
nncudasrcs := $(wildcard $(srcdir)/nn/*.cu)
nnobjs := $(nnsrcs:.c=.o)


.INTERMEDIATE: $(nnobjs)

lib/libnn.a: libnn.a($(nnobjs))


