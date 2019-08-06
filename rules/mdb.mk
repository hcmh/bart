

mdbsrcs := $(wildcard $(srcdir)/mdb/*.c)
mdbobjs := $(mdbsrcs:.c=.o)

.INTERMEDIATE: $(mdbobjs)

lib/libmdb.a: libmdb.a($(mdbobjs))

UTARGETS += test_mdb
MODULES_test_mdb += -lmdb -lnlops -llinops



