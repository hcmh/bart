


tests/test-moba-t1: phantom signal fft ones index scale moba looklocker fmac scale nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco.ra		;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra		    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.005 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-sms: phantom signal fft ones index scale moba looklocker fmac scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 circ.ra 		                  		;\
	$(TOOLDIR)/repmat 13 3 circ.ra circ2.ra		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ2.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 8195 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 16 16 16 1 1 1 100 1 1 1 1 1 1 1 3 1 1 psf.ra			;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -M -L -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco.ra		;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/fmac T1.ra circ2.ra masked.ra		 			;\
	$(TOOLDIR)/scale -- 1.12 circ2.ra ref.ra		    			;\
	$(TOOLDIR)/nrmse -t 0.006 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS += tests/test-moba-t1 tests/test-moba-t1-sms


