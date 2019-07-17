


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


tests/test-laplace-gen: ones zeros join laplace flip scale delta saxpy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/ones 2 10 10 o.ra				;\
		$(TOOLDIR)/zeros 2 10 10 z.ra				;\
		$(TOOLDIR)/join 0  o.ra z.ra o.ra z.ra o.ra z.ra j.ra	;\
		$(TOOLDIR)/laplace -n1 -s1 -g j.ra Lg.ra			;\
		$(TOOLDIR)/flip 1 j.ra jf.ra				;\
		$(TOOLDIR)/join 1 j.ra jf.ra j.ra jf.ra j.ra jf.ra T.ra	;\
		$(TOOLDIR)/scale -- 3.333e-2 T.ra T1.ra			;\
		$(TOOLDIR)/nrmse -t 0.002 T1.ra Lg.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	
tests/test-laplace-nn: phantom ones transpose flip join reshape laplace scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/phantom -x5 -k k0.ra 					;\
		$(TOOLDIR)/transpose 0 1 k0.ra k1.ra					;\
		$(TOOLDIR)/flip 2 k0.ra k2.ra						;\
		$(TOOLDIR)/join 10 k0.ra k0.ra k1.ra k0.ra k2.ra k1.ra k2.ra k.ra	;\
		$(TOOLDIR)/reshape 3 1 25 k.ra krs1.ra					;\
		$(TOOLDIR)/transpose 0 10 krs1.ra ac.ra					;\
		$(TOOLDIR)/laplace -n2 ac.ra L.ra					;\
		$(TOOLDIR)/ones 2 1 1 o1.ra						;\
		$(TOOLDIR)/scale -- -1 o1.ra om1.ra					;\
		$(TOOLDIR)/scale 2 o1.ra o2.ra						;\
		$(TOOLDIR)/scale 0 o1.ra o0.ra						;\
		$(TOOLDIR)/join 1 o2.ra om1.ra o0.ra om1.ra o0.ra o0.ra o0.ra r1.ra	;\
		$(TOOLDIR)/join 1 om1.ra o2.ra o0.ra om1.ra o0.ra o0.ra o0.ra r2.ra	;\
		$(TOOLDIR)/join 1 o0.ra o0.ra o1.ra o0.ra o0.ra om1.ra o0.ra r3.ra	;\
		$(TOOLDIR)/join 1 om1.ra om1.ra o0.ra o2.ra o0.ra o0.ra o0.ra r4.ra	;\
		$(TOOLDIR)/join 1 o0.ra o0.ra o0.ra o0.ra o1.ra o0.ra om1.ra r5.ra	;\
		$(TOOLDIR)/join 1 o0.ra o0.ra om1.ra o0.ra o0.ra o1.ra o0.ra r6.ra	;\
		$(TOOLDIR)/join 1 o0.ra o0.ra o0.ra o0.ra om1.ra o0.ra o1.ra r7.ra	;\
		$(TOOLDIR)/join 0 r1.ra r2.ra r3.ra r4.ra r5.ra r6.ra r7.ra r.ra	;\
		$(TOOLDIR)/nrmse -t 0.00001 r.ra L.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	
	


TESTS += tests/test-laplace tests/test-laplace-gen tests/test-laplace-nn
