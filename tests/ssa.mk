


tests/test-ssa-pca: traj phantom resize squeeze svd transpose ssa cabs nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra				;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra					;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra					;\
		$(TOOLDIR)/squeeze k1.ra kx.ra						;\
		$(TOOLDIR)/svd kx.ra u.ra s.ra vh.ra					;\
		$(TOOLDIR)/transpose 2 10 k1.ra k2.ra					;\
		$(TOOLDIR)/ssa -w1 -m0 -n0 -t0 k2.ra eof.ra				;\
		$(TOOLDIR)/cabs u.ra uabs.ra						;\
		$(TOOLDIR)/cabs eof.ra eofabs.ra					;\
		$(TOOLDIR)/resize 1 4 uabs.ra utest.ra					;\
		$(TOOLDIR)/resize 1 4 eofabs.ra eoftest.ra				;\
		$(TOOLDIR)/nrmse -t 0.00001 utest.ra eoftest.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ssa: traj phantom resize squeeze svd transpose ssa cabs nrmse casorati
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra				;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra					;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra					;\
		$(TOOLDIR)/squeeze k1.ra kx.ra						;\
		$(TOOLDIR)/resize -c 0 59 kx.ra kx1.ra					;\
		$(TOOLDIR)/casorati 0 10 1 8 kx1.ra kcas.ra				;\
		$(TOOLDIR)/svd kcas.ra u.ra s.ra vh.ra					;\
		$(TOOLDIR)/transpose 2 10 k1.ra k2.ra					;\
		$(TOOLDIR)/ssa -w10 -m0 -n0 -t0 k2.ra eof.ra				;\
		$(TOOLDIR)/cabs u.ra uabs.ra						;\
		$(TOOLDIR)/cabs eof.ra eofabs.ra					;\
		$(TOOLDIR)/resize 1 10 uabs.ra utest.ra					;\
		$(TOOLDIR)/resize 1 10 eofabs.ra eoftest.ra				;\
		$(TOOLDIR)/nrmse -t 0.00001 utest.ra eoftest.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-ssa-pca tests/test-ssa
