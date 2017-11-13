
nlopssrcs := $(wildcard $(srcdir)/nlops/*.c)
nlopscudasrcs := $(wildcard $(srcdir)/nlops/*.cu)
nlopsobjs := $(nlopssrcs:.c=.o)


.INTERMEDIATE: $(nlopsobjs)

lib/libnlops.a: libnlops.a($(nlopsobjs))


UTARGETS += test_nlop_zexp
MODULES_test_nlop_zexp += -lnlops -llinops

