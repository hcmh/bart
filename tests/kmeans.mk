


tests/test-kmeans: ones scale zeros join kmeans nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
		$(TOOLDIR)/ones 2 1 1 o.ra					;\
		$(TOOLDIR)/scale -- -1 o.ra om.ra 				;\
		$(TOOLDIR)/zeros 2 2 1 p4.ra 					;\
		$(TOOLDIR)/join 0 o.ra o.ra p0.ra 				;\
		$(TOOLDIR)/join 0 o.ra om.ra p1.ra				;\
		$(TOOLDIR)/join 0 om.ra om.ra p2.ra				;\
		$(TOOLDIR)/join 0 om.ra o.ra p3.ra 				;\
		$(TOOLDIR)/join 1 p0.ra p1.ra p0.ra p2.ra p1.ra p3.ra p4.ra p4.ra p3.ra p2.ra p0.ra p4.ra p3.ra p.ra	;\
		$(TOOLDIR)/squeeze p.ra pp.ra	;\
		$(TOOLDIR)/join 1 p0.ra p4.ra p1.ra p2.ra p3.ra cen.ra		;\
		$(TOOLDIR)/ones 1 1 o.ra 					;\
		$(TOOLDIR)/scale 0 o.ra o0.ra					;\
		$(TOOLDIR)/scale 1 o.ra o1.ra					;\
		$(TOOLDIR)/scale 2 o.ra o2.ra					;\
		$(TOOLDIR)/scale 3 o.ra o3.ra					;\
		$(TOOLDIR)/scale 4 o.ra o4.ra					;\
		$(TOOLDIR)/join 0 o0.ra o2.ra o0.ra o3.ra o2.ra o4.ra o1.ra o1.ra o4.ra o3.ra o0.ra o1.ra o4.ra lab.ra	;\
		$(TOOLDIR)/kmeans -k5 pp.ra res_cen.ra res_lab.ra		;\
		$(TOOLDIR)/nrmse -t 0.0001 lab.ra res_lab.ra			;\
		$(TOOLDIR)/nrmse -t 0.0001 cen.ra res_cen.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-kmeans
