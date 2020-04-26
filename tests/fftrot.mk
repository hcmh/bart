



tests/test-fftrot0: fftrot flip transpose circshift nrmse $(TESTS_OUT)/shepplogan.ra 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/fftrot 0 1 90 $(TESTS_OUT)/shepplogan.ra x.ra	;\
	$(TOOLDIR)/flip 1 $(TESTS_OUT)/shepplogan.ra tmp.ra		;\
	$(TOOLDIR)/transpose 0 1 tmp.ra tmp1.ra				;\
	$(TOOLDIR)/circshift 1 1 tmp1.ra ref.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 ref.ra x.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-fftrot1: fftrot nrmse $(TESTS_OUT)/shepplogan.ra 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/fftrot 0 1 33. $(TESTS_OUT)/shepplogan.ra tmp.ra	;\
	$(TOOLDIR)/fftrot -- 0 1 -33. tmp.ra x.ra			;\
	$(TOOLDIR)/nrmse -t 0.000001  $(TESTS_OUT)/shepplogan.ra x.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-fftrot2: copy fftrot nrmse $(TESTS_OUT)/shepplogan.ra 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/copy $(TESTS_OUT)/shepplogan.ra ref.ra			;\
	$(TOOLDIR)/fftrot 0 1 90 ref.ra x1.ra					;\
	$(TOOLDIR)/fftrot 0 1 90 x1.ra x0.ra					;\
	$(TOOLDIR)/fftrot 0 1 90 x0.ra x1.ra					;\
	$(TOOLDIR)/fftrot 0 1 90 x1.ra x0.ra					;\
	$(TOOLDIR)/nrmse -t 0.000005 ref.ra x0.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-fftrot3: resize fftrot nrmse $(TESTS_OUT)/shepplogan.ra 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/resize -c 0 140 1 140 $(TESTS_OUT)/shepplogan.ra ref.ra	;\
	$(TOOLDIR)/fftrot 0 1 60 ref.ra x1.ra					;\
	$(TOOLDIR)/fftrot 0 1 60 x1.ra x0.ra					;\
	$(TOOLDIR)/fftrot 0 1 60 x0.ra x1.ra					;\
	$(TOOLDIR)/fftrot 0 1 60 x1.ra x0.ra					;\
	$(TOOLDIR)/fftrot 0 1 60 x0.ra x1.ra					;\
	$(TOOLDIR)/fftrot 0 1 60 x1.ra x0.ra					;\
	$(TOOLDIR)/nrmse -t 0.2 ref.ra x0.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS += tests/test-fftrot0 tests/test-fftrot1 tests/test-fftrot2 tests/test-fftrot3

