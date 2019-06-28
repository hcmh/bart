tests/test-bin: ones scale zeros join transpose bin resize nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/ones 2 1 1 o.ra						;\
		$(TOOLDIR)/scale -- -1 o.ra om.ra					;\
		$(TOOLDIR)/zeros 2 2 1 p4.ra						;\
		$(TOOLDIR)/join 0 o.ra o.ra p0.ra					;\
		$(TOOLDIR)/join 0 o.ra om.ra p1.ra					;\
		$(TOOLDIR)/join 0 om.ra om.ra p2.ra					;\
		$(TOOLDIR)/join 0 om.ra o.ra p3.ra					;\
		$(TOOLDIR)/join 1 p0.ra p1.ra p0.ra p2.ra p1.ra p3.ra p4.ra p4.ra p3.ra p2.ra p0.ra p4.ra p3.ra p.ra	;\
		$(TOOLDIR)/ones 1 1 o.ra						;\
		$(TOOLDIR)/scale 0 o.ra o0.ra						;\
		$(TOOLDIR)/scale 1 o.ra o1.ra						;\
		$(TOOLDIR)/scale 2 o.ra o2.ra						;\
		$(TOOLDIR)/scale 3 o.ra o3.ra						;\
		$(TOOLDIR)/scale 4 o.ra o4.ra						;\
		$(TOOLDIR)/join 0 o0.ra o2.ra o0.ra o3.ra o2.ra o4.ra o1.ra o1.ra o4.ra o3.ra o0.ra o1.ra o4.ra lab.ra	;\
		$(TOOLDIR)/transpose 0 1 lab.ra lab1.ra						;\
		$(TOOLDIR)/bin -d2 lab1.ra p.ra x.ra						;\
		$(TOOLDIR)/join 2 p0.ra p0.ra p0.ra pa.ra					;\
		$(TOOLDIR)/resize 2 3 pa.ra paa.ra						;\
		$(TOOLDIR)/join 2 p1.ra p1.ra pb.ra						;\
		$(TOOLDIR)/resize 2 3 pb.ra pbb.ra						;\
		$(TOOLDIR)/join 2 p2.ra p2.ra pc.ra						;\
		$(TOOLDIR)/resize 2 3 pc.ra pcc.ra						;\
		$(TOOLDIR)/join 2 p3.ra p3.ra p3.ra pd.ra					;\
		$(TOOLDIR)/resize 2 3 pd.ra pdd.ra						;\
		$(TOOLDIR)/join 2 p4.ra p4.ra pe.ra						;\
		$(TOOLDIR)/resize 2 3 pe.ra pee.ra						;\
		$(TOOLDIR)/join 1 paa.ra pee.ra pbb.ra pcc.ra pdd.ra comp.ra			;\
		$(TOOLDIR)/nrmse -t 0.00001 comp.ra x.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-bin
