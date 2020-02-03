

tests/test-cordelay: traj phantom cordelay nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x128 -y127 -G t.ra					;\
		$(TOOLDIR)/traj -x128 -y127 -G -q 2:-1:0 to.ra				;\
		$(TOOLDIR)/phantom -t t.ra -s2 k.ra					;\
		$(TOOLDIR)/phantom -t to.ra -s2 ko.ra					;\
		$(TOOLDIR)/cordelay -q 2:-1:0 to.ra ko.ra  k_cor.ra			;\
		$(TOOLDIR)/nrmse -t 0.005 k_cor.ra k.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-cordelay-B0: traj phantom estdelay cordelay nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x4 -y91 -r -G -s7 -q0.2:0.2:0 -c t.ra			;\
		$(TOOLDIR)/phantom -k -t t.ra -s7 k.ra					;\
		$(TOOLDIR)/traj -x4 -y91 -r -G -s7 -c t0.ra				;\
		$(TOOLDIR)/phantom -k -t t0.ra -s7 k0.ra      				;\
		$(TOOLDIR)/carg k0.ra k0arg.ra						;\
		$(TOOLDIR)/estdelay -B t0.ra k.ra G.ra					;\
		$(TOOLDIR)/cordelay -B G t.ra k.ra kcor.ra				;\
		$(TOOLDIR)/carg kcor.ra kcorarg.ra					;\
		$(TOOLDIR)/nrmse -t 0.7 k0arg.ra kcorarg.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	

TESTS += tests/test-cordelay
