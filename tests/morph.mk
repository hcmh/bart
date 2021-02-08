
tests/test-morph-dilation-erosion: phantom morph nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -g 3 ori.ra					;\
	$(TOOLDIR)/morph -e -b 9 ori.ra redu.ra					;\
	$(TOOLDIR)/morph -d -b 9 redu.ra rec.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ori.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-morph-dilation-erosion

