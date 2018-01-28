
nlopssrcs := $(wildcard $(srcdir)/nlops/*.c)
nlopscudasrcs := $(wildcard $(srcdir)/nlops/*.cu)
nlopsobjs := $(nlopssrcs:.c=.o)


.INTERMEDIATE: $(nlopsobjs)

lib/libnlops.a: libnlops.a($(nlopsobjs))


UTARGETS += test_nlop_zexp test_nlop_chain test_nlop_cast test_nlop_tenmul
MODULES_test_nlop_zexp += -lnlops -llinops
MODULES_test_nlop_tenmul += -lnlops -llinops
MODULES_test_nlop_chain += -lnlops -llinops
MODULES_test_nlop_cast += -lnlops -llinops

