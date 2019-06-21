


tests/test-laplace: ones zeros join laplace flip scale delta saxpy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/ones 2 10 10 o.ra				;\
		$(TOOLDIR)/zeros 2 10 10 z.ra				;\
		$(TOOLDIR)/join 0  o.ra z.ra o.ra z.ra o.ra z.ra j.ra	;\
		$(TOOLDIR)/laplace -n1 -s1 j.ra L.ra			;\
		$(TOOLDIR)/flip 1 j.ra jf.ra				;\
		$(TOOLDIR)/join 1 j.ra jf.ra j.ra jf.ra j.ra jf.ra T.ra	;\
		$(TOOLDIR)/scale -- -1 T.ra T1.ra			;\
		$(TOOLDIR)/delta 2 3 60 delta.ra			;\
		$(TOOLDIR)/saxpy 30 delta.ra T1.ra T2.ra		;\
		$(TOOLDIR)/nrmse -t 0.0001 L.ra T2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-laplace
