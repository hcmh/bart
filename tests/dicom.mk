
tests/test-dicom: phantom toimg dcmread nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/phantom ph.ra				;\
	$(TOOLDIR)/toimg ph.ra ph.dcm				;\
	$(TOOLDIR)/dcmread ph.dcm ph2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 ph.ra ph2.ra			;\
	rm *.dcm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-dicom

