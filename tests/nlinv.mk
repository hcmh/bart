


tests/test-nlinv: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra			;\
	$(TOOLDIR)/normalize 8 c.ra c_norm.ra						;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-sms: repmat fft nlinv nrmse scale $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 13 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra		;\
	$(TOOLDIR)/fft 8192 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra			;\
	$(TOOLDIR)/nlinv ksp2.ra r2.ra							;\
	$(TOOLDIR)/repmat 13 4 r.ra r3.ra						;\
	$(TOOLDIR)/scale 2. r2.ra r4.ra							;\
	$(TOOLDIR)/nrmse -s -t 0.1 r3.ra r4.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-gpu: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv -g $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/normalize 8 c.ra c_norm.ra						;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-sms-gpu: repmat fft nlinv nrmse scale $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 13 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra		;\
	$(TOOLDIR)/fft 8192 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/nlinv -g $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra			;\
	$(TOOLDIR)/nlinv -g ksp2.ra r2.ra						;\
	$(TOOLDIR)/repmat 13 4 r.ra r3.ra						;\
	$(TOOLDIR)/scale 2. r2.ra r4.ra							;\
	$(TOOLDIR)/nrmse -s -t 0.1 r3.ra r4.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-nlinv tests/test-nlinv-sms
TESTS_GPU += tests/test-nlinv-gpu tests/test-nlinv-sms-gpu
