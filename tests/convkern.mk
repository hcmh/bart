


tests/test-convkern-sobel: ones scale join convkern nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/ones 2 1 1 o.ra				;\
	$(TOOLDIR)/scale 2 o.ra o2.ra				;\
	$(TOOLDIR)/join 1 o.ra o2.ra o.ra lp.ra			;\
	$(TOOLDIR)/scale 0 lp.ra l0.ra				;\
	$(TOOLDIR)/scale -- -1 lp.ra lm.ra			;\
	$(TOOLDIR)/join 0 lm.ra l0.ra lp.ra sobel.ra		;\
	$(TOOLDIR)/convkern -s 2 s.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 s.ra sobel.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-convkern-gauss: ones scale join convkern nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/ones 2 1 1 o.ra				;\
	$(TOOLDIR)/scale 0.2042 o.ra o1.ra			;\
        $(TOOLDIR)/scale 0.1238 o.ra o2.ra			;\
	$(TOOLDIR)/scale 0.07511 o.ra o3.ra			;\
	$(TOOLDIR)/join 1 o3.ra o2.ra o3.ra l0.ra		;\
	$(TOOLDIR)/join 1 o2.ra o1.ra o2.ra l1.ra		;\
	$(TOOLDIR)/join 0 l0.ra l1.ra l0.ra gauss.ra		;\
	$(TOOLDIR)/convkern -g 3:1 2 g.ra			;\
	$(TOOLDIR)/nrmse -t 0.0003 g.ra gauss.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@	


TESTS += tests/test-convkern-sobel tests/test-convkern-gauss
