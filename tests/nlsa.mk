

tests/test-nlsa: traj phantom resize squeeze nlsa cabs nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra						;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra 						;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra 						;\
		$(TOOLDIR)/squeeze k1.ra kx.ra 							;\
		$(TOOLDIR)/nlsa -w10 -m0 -n0 kx.ra eof.ra 					;\
		$(TOOLDIR)/nlsa -L50 -w10 -m0 -n0 kx.ra nlsa.ra 				;\
		$(TOOLDIR)/cabs nlsa.ra nlsaabs.ra						;\
		$(TOOLDIR)/cabs eof.ra eofabs.ra 						;\
		$(TOOLDIR)/resize 1 10 nlsaabs.ra utest.ra 					;\
		$(TOOLDIR)/resize 1 10 eofabs.ra eoftest.ra 					;\
		$(TOOLDIR)/nrmse -t 0.0001 utest.ra eoftest.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlsa-backprojection: traj phantom resize squeeze nlsa nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) 						;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra 						;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra 						;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra 						;\
		$(TOOLDIR)/squeeze k1.ra kx.ra 							;\
		$(TOOLDIR)/nlsa -L40 -r40 -w10 -m0 -z -n0 kx.ra nlsa.ra s.ra back.ra 		;\
		$(TOOLDIR)/nrmse -t 0.00001 kx.ra back.ra 	 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlsa-grouping: traj phantom resize squeeze nlsa nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra					;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra					;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra					;\
		$(TOOLDIR)/squeeze k1.ra kx.ra						;\
		$(TOOLDIR)/nlsa -L8 -w10 -m0 -n0 -z -r -5 kx.ra eof.ra s.ra backr.ra	;\
		$(TOOLDIR)/nlsa -L8 -w10 -m0 -n0 -z -g -31 kx.ra eofg.ra sg.ra backg.ra ;\
		$(TOOLDIR)/nrmse -t 0.00001 backr.ra backg.ra				;\
		$(TOOLDIR)/nlsa -L8 -w10 -m0 -n0 -z -r 5 kx.ra eof1.ra s1.ra backr1.ra ;\
		$(TOOLDIR)/nlsa -L8 -w10 -m0 -n0 -z -g 31 kx.ra eofg1.ra sg1.ra backg1.ra ;\
		$(TOOLDIR)/nrmse -t 0.00001 backr1.ra backg1.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-nlsa tests/test-nlsa-backprojection tests/test-nlsa-grouping
