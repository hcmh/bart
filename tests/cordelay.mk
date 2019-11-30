

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


TESTS += tests/test-cordelay
