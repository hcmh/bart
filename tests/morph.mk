
tests/test-morph-dilation-erosion: phantom morph nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -g 3 ori.ra					;\
	$(TOOLDIR)/morph -e -b 9 ori.ra redu.ra					;\
	$(TOOLDIR)/morph -d -b 9 redu.ra rec.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ori.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-morph-dilation-erosion-large: phantom morph nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x128 -g 3 ori.ra					;\
	$(TOOLDIR)/morph -e -b 51 ori.ra redu.ra					;\
	$(TOOLDIR)/morph -d -b 51 redu.ra rec.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ori.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-morph-opening: phantom morph nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -g 2 ori.ra					;\
	$(TOOLDIR)/morph -e -b 9 ori.ra tmp.ra					;\
	$(TOOLDIR)/morph -d -b 9 tmp.ra rec.ra					;\
	$(TOOLDIR)/morph -o -b 9 ori.ra rec2.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 rec2.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-morph-closing: phantom morph nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -g 2 ori.ra					;\
	$(TOOLDIR)/morph -d -b 9 ori.ra tmp.ra					;\
	$(TOOLDIR)/morph -e -b 9 tmp.ra rec.ra					;\
	$(TOOLDIR)/morph -c -b 9 ori.ra rec2.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 rec2.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-morph-dilation-erosion tests/test-morph-dilation-erosion-large tests/test-morph-opening tests/test-morph-closing

