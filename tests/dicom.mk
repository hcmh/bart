
tests/test-dicom: phantom toimg dcmread nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/phantom ph.ra				;\
	$(TOOLDIR)/toimg ph.ra ph.dcm				;\
	$(TOOLDIR)/dcmread ph.dcm ph2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 ph.ra ph2.ra			;\
	rm *.dcm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-dicom-tag: zeros toimg dcmtag
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/zeros 2 100 103 x.ra				;\
	$(TOOLDIR)/toimg x.ra x.dcm				;\
	[ 100 = `$(TOOLDIR)/dcmtag 0028,0010 x.dcm` ]		;\
	[ 103 = `$(TOOLDIR)/dcmtag 0028,0011 x.dcm` ]		;\
	rm *.dcm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-dicom tests/test-dicom-tag

