


tests/test-moba-t1: phantom signal fft ones index scale moba looklocker fmac nrmse 
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


tests/test-moba-t1-magn: phantom signal fft ones index scale moba normalize slice fmac nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/normalize 8 sens.ra norm.ra						;\
	$(TOOLDIR)/slice 6 0 reco.ra magn.ra						;\
	$(TOOLDIR)/fmac magn.ra norm.ra reco2.ra		    			;\
	$(TOOLDIR)/nrmse -s -t 0.001 circ.ra reco2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-t1-sms: phantom signal fft ones index scale moba looklocker fmac nrmse
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


tests/test-moba-t1-no-IR: phantom signal fft ones index scale moba looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -F -r0.005 -n300 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 300 psf.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco.ra		;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra		    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.005 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-nonCartesian: traj transpose phantom signal nufft fft ones index scale moba looklocker resize fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y1 -r -D -G -s7 -t300 _traj.ra  		                ;\
	$(TOOLDIR)/transpose 5 10 _traj.ra _traj2.ra    		                ;\
	$(TOOLDIR)/scale 0.5 _traj2.ra traj.ra   	    		                ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra    	    		        ;\
	$(TOOLDIR)/signal -F -I -r0.005 -n300 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra				;\
 	$(TOOLDIR)/ones 16 1 16 1 1 1 300 1 1 1 1 1 1 1 1 1 1 ones.ra	   		;\
	$(TOOLDIR)/nufft -d 16:16:1 -a _traj2.ra ones.ra pattern.ra	   		;\
	$(TOOLDIR)/fft -u 3 pattern.ra psf.ra				   		;\
	$(TOOLDIR)/nufft -d 16:16:1 -a _traj2.ra data.ra zerofill_reco.ra  		;\
	$(TOOLDIR)/fft -u 3 zerofill_reco.ra k_space.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -l1 -i11 -C200 -j0.01 -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra		    			;\
	$(TOOLDIR)/resize -c 0 8 1 8 T1.ra T1_crop.ra					;\
	$(TOOLDIR)/phantom -x8 -c circ.ra						;\
	$(TOOLDIR)/fmac T1_crop.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.02 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-nufft: traj transpose phantom signal nufft fft ones index scale moba fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y1 -r -D -G -s7 -t300 _traj.ra  		                ;\
	$(TOOLDIR)/transpose 5 10 _traj.ra _traj2.ra    		                ;\
	$(TOOLDIR)/scale 0.5 _traj2.ra traj.ra   	    		                ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra    	    		        ;\
	$(TOOLDIR)/signal -F -I -r0.005 -n300 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra				;\
 	$(TOOLDIR)/ones 16 1 16 1 1 1 300 1 1 1 1 1 1 1 1 1 1 ones.ra	   		;\
	$(TOOLDIR)/nufft -d 16:16:1 -a _traj2.ra ones.ra pattern.ra	   		;\
	$(TOOLDIR)/fft -u 3 pattern.ra psf.ra				   		;\
	$(TOOLDIR)/nufft -d 16:16:1 -a _traj2.ra data.ra zerofill_reco.ra  		;\
	$(TOOLDIR)/fft -u 3 zerofill_reco.ra k_space.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -l1 -i11 -C200 -j0.01 -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/moba -L -l1 -i11 -C200 -j0.01 -o1.0 -t _traj2.ra data.ra TI.ra reco2.ra ;\
	$(TOOLDIR)/nrmse -t 0.00003 reco.ra reco2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-MOLLI: phantom signal fft ones index scale moba slice invert fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -M -I -r0.004 -n300 -1 1.12:1.12:1 -2 100:100:1 -t0.36 -b5 signal.ra ;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 300 psf.ra					;\
	$(TOOLDIR)/index 5 300 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.004 tmp1.ra TI.ra            	       			;\
	$(TOOLDIR)/ones 1 4 tmp2.ra   					                ;\
	$(TOOLDIR)/scale 0.36 tmp2.ra t1relax.ra            	       			;\
	$(TOOLDIR)/moba -L -m -i11 -f1 -C200 -T t1relax.ra -p psf.ra k_space.ra TI.ra reco.ra ;\
	$(TOOLDIR)/slice 6 1 reco.ra R1.ra						;\
	$(TOOLDIR)/invert R1.ra T1.ra							;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra						;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra						;\
	$(TOOLDIR)/nrmse -t 0.00007 masked.ra ref.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-irgnm-admm: phantom signal fmac fft ones index scale moba looklocker nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -n600 -1 1.12:1.12:1 -2 100:100:1 signal.ra     ;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 600 psf.ra					;\
	$(TOOLDIR)/index 5 600 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                  	       			;\
	$(TOOLDIR)/moba -L -l3 -i11 -f1 -p psf.ra k_space.ra TI.ra reco.ra	        ;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra					;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra						;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra						;\
	$(TOOLDIR)/nrmse -t 0.0008 masked.ra ref.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-moba-t1 tests/test-moba-t1-sms tests/test-moba-t1-no-IR
TESTS += tests/test-moba-t1-magn tests/test-moba-t1-nonCartesian tests/test-moba-t1-nufft
TESTS += tests/test-moba-t1-MOLLI
TESTS += tests/test-moba-t1-irgnm-admm

