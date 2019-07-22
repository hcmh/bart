# Copyright 2017. Martin Uecker.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




geomsrcs := $(wildcard $(srcdir)/geom/*.c)
geomobjs := $(geomsrcs:.c=.o)

.INTERMEDIATE: $(geomobjs)

lib/libgeom.a: libgeom.a($(geomobjs))


UTARGETS += test_geom
MODULES_test_geom += -lgeom

