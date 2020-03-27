
tests/test-pad-val-asym: ones scale join pad nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 2 1 o.ra					;\
	$(TOOLDIR)/scale 0.5 o.ra o5.ra				;\
	$(TOOLDIR)/join 1 o.ra o5.ra j.ra			;\
	$(TOOLDIR)/pad -a 1 3 1+2i j.ra j_pad.ra    ;\
	$(TOOLDIR)/scale 1+2i o.ra os.ra			;\
	$(TOOLDIR)/join 1 j.ra os.ra os.ra os.ra j_syn.ra ;\
	$(TOOLDIR)/nrmse -t 0.0 j_syn.ra j_pad.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	
tests/test-pad-val-sym: ones scale join pad nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 2 1 o.ra					;\
	$(TOOLDIR)/scale 0.5 o.ra o5.ra				;\
	$(TOOLDIR)/join 1 o.ra o5.ra j.ra			;\
	$(TOOLDIR)/pad 1 3 1+2i j.ra j_pad.ra    ;\
	$(TOOLDIR)/scale 1+2i o.ra os.ra			;\
	$(TOOLDIR)/join 1 os.ra os.ra os.ra j.ra os.ra os.ra os.ra j_syn.ra ;\
	$(TOOLDIR)/nrmse -t 0.0 j_syn.ra j_pad.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pad-edge-asym: ones scale join pad nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 2 1 o.ra					;\
	$(TOOLDIR)/scale 0.5 o.ra o5.ra				;\
	$(TOOLDIR)/join 1 o.ra o5.ra j.ra			;\
	$(TOOLDIR)/scale 0.5 j.ra jj.ra				;\
	$(TOOLDIR)/join 0 j.ra jj.ra jjj.ra			;\
	$(TOOLDIR)/pad -a 0 4 jjj.ra j_pad.ra	;\
	$(TOOLDIR)/join 0 jjj.ra jj.ra jj.ra j_syn.ra ;\
	$(TOOLDIR)/nrmse -t 0.0 j_syn.ra j_pad.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
				
tests/test-pad-edge-sym: ones scale join pad nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 2 1 o.ra					;\
	$(TOOLDIR)/scale 0.5 o.ra o5.ra				;\
	$(TOOLDIR)/join 1 o.ra o5.ra j.ra			;\
	$(TOOLDIR)/scale 0.5 j.ra jj.ra				;\
	$(TOOLDIR)/join 0 j.ra jj.ra jjj.ra			;\
	$(TOOLDIR)/pad  0 4 jjj.ra j_pad.ra    ;\
	$(TOOLDIR)/join 0 j.ra j.ra jjj.ra jj.ra jj.ra j_syn.ra		;\
	$(TOOLDIR)/nrmse -t 0.0 j_syn.ra j_pad.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
				

TESTS += tests/test-pad-val-asym tests/test-pad-val-sym tests/test-pad-edge-asym tests/test-pad-edge-sym

