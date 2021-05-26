


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
	$(TOOLDIR)/moba --irflash L -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
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
	$(TOOLDIR)/moba --irflash L -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco2.ra sens2.ra	;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
	$(TOOLDIR)/nrmse -t 0.00001 sens2.ra sens.ra			    		;\
	$(TOOLDIR)/normalize 8 sens.ra norm.ra						;\
	$(TOOLDIR)/slice 6 0 reco.ra magn.ra						;\
	$(TOOLDIR)/fmac magn.ra norm.ra reco2.ra		    			;\
	$(TOOLDIR)/nrmse -s -t 0.001 circ.ra reco2.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-sms: phantom signal repmat fft ones index scale moba looklocker fmac nrmse
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
	$(TOOLDIR)/moba --irflash L -M -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
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
	$(TOOLDIR)/moba --irflash L -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
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
	$(TOOLDIR)/moba --irflash L -l1 -i11 -C200 -j0.01 -p psf.ra k_space.ra TI.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
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
	$(TOOLDIR)/moba --irflash L -l1 -i11 -C200 -j0.01 -o1.0 -t _traj2.ra data.ra TI.ra reco3.ra ;\
	$(TOOLDIR)/nrmse -t 0.00004 reco.ra reco2.ra			    		;\
	$(TOOLDIR)/nrmse -t 0.00004 reco2.ra reco3.ra			    		;\
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
	$(TOOLDIR)/moba --irflash M -i11 -f1 -C200 -T t1relax.ra -p psf.ra k_space.ra TI.ra reco2.ra ;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
	$(TOOLDIR)/slice 6 1 reco.ra R1.ra						;\
	$(TOOLDIR)/invert R1.ra T1.ra							;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra						;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra						;\
	$(TOOLDIR)/nrmse -t 0.00007 masked.ra ref.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-IR_SS: phantom signal fft ones index scale moba slice fmac nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -s -F -r0.005 -n100 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 100 psf.ra					;\
	$(TOOLDIR)/index 5 100 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba -L -S -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco.ra		;\
	$(TOOLDIR)/moba --irflash S -i11 -f1 -C200 -p psf.ra k_space.ra TI.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
	$(TOOLDIR)/slice 6 1 reco.ra R1s.ra						;\
	$(TOOLDIR)/fmac R1s.ra circ.ra masked.ra		    			;\
	$(TOOLDIR)/scale -- 2.848 circ.ra ref.ra			    		;\
	$(TOOLDIR)/nrmse -t 0.0005 masked.ra ref.ra			    		;\
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
	$(TOOLDIR)/moba -L -rW:3:0:0.1 -i11 -f1 -p psf.ra k_space.ra TI.ra reco.ra      ;\
	$(TOOLDIR)/moba --irflash L -rW:3:0:0.1 -i11 -f1 -p psf.ra k_space.ra TI.ra reco2.ra      ;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
	$(TOOLDIR)/looklocker -t0.1 -D0. reco.ra T1.ra					;\
	$(TOOLDIR)/fmac T1.ra circ.ra masked.ra						;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra						;\
	$(TOOLDIR)/nrmse -t 0.001 masked.ra ref.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t2: phantom signal fmac fft ones index scale moba slice invert nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/phantom -x16 -c circ.ra 		                  		;\
	$(TOOLDIR)/signal -T -e0.01 -n16 -1 1.25:1.25:1 -2 0.09:0.09:1 signal.ra  	;\
	$(TOOLDIR)/fmac circ.ra signal.ra image.ra					;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra						;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 16 psf.ra						;\
	$(TOOLDIR)/index 5 16 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.01 tmp1.ra TE.ra                  	       			;\
	$(TOOLDIR)/moba -F -i10 -f1 -C200 -d4 -p psf.ra k_space.ra TE.ra reco.ra	;\
	$(TOOLDIR)/moba --spin-echo F -i10 -f1 -C200 -d4 -p psf.ra k_space.ra TE.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.00001 reco2.ra reco.ra			    		;\
	$(TOOLDIR)/slice 6 1 reco.ra R2.ra						;\
	$(TOOLDIR)/invert R2.ra T2.ra							;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac T2.ra circ.ra masked.ra						;\
	$(TOOLDIR)/scale -- 0.9 circ.ra ref.ra						;\
	$(TOOLDIR)/nrmse -t 0.0008 masked.ra ref.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-meco-noncart-r2s: traj scale phantom signal fmac index extract moba slice resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                  ;\
	$(TOOLDIR)/traj -x16 -y15 -r -D -E -e7 -c _traj.ra                ;\
	$(TOOLDIR)/scale 0.5 _traj.ra traj.ra                             ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra                 ;\
	$(TOOLDIR)/signal -G -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra     ;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra                   ;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra             ;\
	$(TOOLDIR)/index 5 8 tmp1.ra                                      ;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra                              ;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra                            ;\
	$(TOOLDIR)/moba -G -D3 -rQ:1 -rS:0 -rW:3:64:1 -i10 -C100 -u0.0001 -R3 -o1.5 -k --kfilter-2 -t _traj.ra data.ra TE.ra reco.ra   ;\
	$(TOOLDIR)/moba --multi-gre M -D3 -rQ:1 -rS:0 -rW:3:64:1 -i10 -C100 -u0.0001 -R3 -o1.5 -k --kfilter-2 -t _traj.ra data.ra TE.ra reco2.ra   ;\
	$(TOOLDIR)/nrmse -t 0.008 reco2.ra reco.ra			  ;\
	$(TOOLDIR)/slice 6 1 reco.ra R2S.ra                               ;\
	$(TOOLDIR)/resize -c 0 8 1 8 R2S.ra R2S_crop.ra                   ;\
	$(TOOLDIR)/phantom -x8 -c circ.ra                                 ;\
	$(TOOLDIR)/fmac R2S_crop.ra circ.ra masked.ra                     ;\
	$(TOOLDIR)/scale -- 50 circ.ra ref.ra                             ;\
	$(TOOLDIR)/nrmse -t 0.008 ref.ra masked.ra                        ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-meco-noncart-wfr2s: traj scale phantom signal fmac index extract moba slice resize saxpy cabs spow nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                  ;\
	$(TOOLDIR)/traj -x16 -y15 -r -D -E -e7 -c _traj.ra                ;\
	$(TOOLDIR)/scale 0.5 _traj.ra traj.ra                             ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra                 ;\
	$(TOOLDIR)/signal -G --fat -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra  ;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra                   ;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra             ;\
	$(TOOLDIR)/index 5 8 tmp1.ra                                      ;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra                              ;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra                            ;\
	$(TOOLDIR)/moba -G -D1 -rQ:1 -rS:0 -rW:3:64:1 -i10 -C100 -u0.0001 -R3 -o1.5 -k --kfilter-2 -t _traj.ra data.ra TE.ra reco.ra   ;\
	$(TOOLDIR)/resize -c 0 8 1 8 reco.ra reco_crop.ra                 ;\
	$(TOOLDIR)/slice 6 0 reco_crop.ra W.ra                            ;\
	$(TOOLDIR)/slice 6 1 reco_crop.ra F.ra                            ;\
	$(TOOLDIR)/slice 6 2 reco_crop.ra R2S.ra                          ;\
	$(TOOLDIR)/slice 6 3 reco_crop.ra fB0.ra                          ;\
	$(TOOLDIR)/saxpy 1 W.ra F.ra temp_inphase.ra                      ;\
	$(TOOLDIR)/cabs temp_inphase.ra temp_inphase_abs.ra               ;\
	$(TOOLDIR)/spow -- -1. temp_inphase_abs.ra temp_deno.ra           ;\
	$(TOOLDIR)/cabs F.ra temp_F_abs.ra                                ;\
	$(TOOLDIR)/fmac temp_F_abs.ra temp_deno.ra fatfrac.ra             ;\
	$(TOOLDIR)/phantom -x8 -c circ.ra                                 ;\
	$(TOOLDIR)/fmac fatfrac.ra circ.ra fatfrac_masked.ra              ;\
	$(TOOLDIR)/scale -- 0.20 circ.ra fatfrac_ref.ra                   ;\
	$(TOOLDIR)/nrmse -t 0.02 fatfrac_ref.ra fatfrac_masked.ra         ;\
	$(TOOLDIR)/fmac R2S.ra circ.ra R2S_masked.ra                      ;\
	$(TOOLDIR)/scale -- 50 circ.ra R2S_ref.ra                         ;\
	$(TOOLDIR)/nrmse -t 0.008 R2S_ref.ra R2S_masked.ra                ;\
	$(TOOLDIR)/fmac fB0.ra circ.ra fB0_masked.ra                      ;\
	$(TOOLDIR)/scale -- 20 circ.ra fB0_ref.ra                         ;\
	$(TOOLDIR)/nrmse -t 0.0003 fB0_ref.ra fB0_masked.ra               ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-moba-meco-noncart-wfr2s-ui: traj scale phantom signal fmac index extract moba slice resize saxpy cabs spow nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                  ;\
	$(TOOLDIR)/traj -x16 -y15 -r -D -E -e7 -c _traj.ra                ;\
	$(TOOLDIR)/scale 0.5 _traj.ra traj.ra                             ;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra                 ;\
	$(TOOLDIR)/signal -G --fat -n8 -1 3:3:1 -2 0.02:0.02:1 signal_p1.ra  ;\
	$(TOOLDIR)/extract 5 1 8 signal_p1.ra signal.ra                   ;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra signal.ra data.ra             ;\
	$(TOOLDIR)/index 5 8 tmp1.ra                                      ;\
	$(TOOLDIR)/scale 1.6 tmp1.ra tmp2.ra                              ;\
	$(TOOLDIR)/extract 5 1 8 tmp2.ra TE.ra                            ;\
	$(TOOLDIR)/moba --multi-gre M -D1 -rQ:1 -rS:0 -rW:3:64:1 -i10 -C100 -u0.0001 -R3 -o1.5 -k --kfilter-2 -t _traj.ra data.ra TE.ra reco.ra   ;\
	$(TOOLDIR)/resize -c 0 8 1 8 reco.ra reco_crop.ra                 ;\
	$(TOOLDIR)/slice 6 0 reco_crop.ra W.ra                            ;\
	$(TOOLDIR)/slice 6 1 reco_crop.ra F.ra                            ;\
	$(TOOLDIR)/slice 6 2 reco_crop.ra R2S.ra                          ;\
	$(TOOLDIR)/slice 6 3 reco_crop.ra fB0.ra                          ;\
	$(TOOLDIR)/saxpy 1 W.ra F.ra temp_inphase.ra                      ;\
	$(TOOLDIR)/cabs temp_inphase.ra temp_inphase_abs.ra               ;\
	$(TOOLDIR)/spow -- -1. temp_inphase_abs.ra temp_deno.ra           ;\
	$(TOOLDIR)/cabs F.ra temp_F_abs.ra                                ;\
	$(TOOLDIR)/fmac temp_F_abs.ra temp_deno.ra fatfrac.ra             ;\
	$(TOOLDIR)/phantom -x8 -c circ.ra                                 ;\
	$(TOOLDIR)/fmac fatfrac.ra circ.ra fatfrac_masked.ra              ;\
	$(TOOLDIR)/scale -- 0.20 circ.ra fatfrac_ref.ra                   ;\
	$(TOOLDIR)/nrmse -t 0.02 fatfrac_ref.ra fatfrac_masked.ra         ;\
	$(TOOLDIR)/fmac R2S.ra circ.ra R2S_masked.ra                      ;\
	$(TOOLDIR)/scale -- 50 circ.ra R2S_ref.ra                         ;\
	$(TOOLDIR)/nrmse -t 0.008 R2S_ref.ra R2S_masked.ra                ;\
	$(TOOLDIR)/fmac fB0.ra circ.ra fB0_masked.ra                      ;\
	$(TOOLDIR)/scale -- 20 circ.ra fB0_ref.ra                         ;\
	$(TOOLDIR)/nrmse -t 0.0003 fB0_ref.ra fB0_masked.ra               ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-alpha-non-cartesian: traj repmat phantom signal fmac index scale moba slice spow nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x12 -y12 _traj.ra                  ;\
	$(TOOLDIR)/repmat 5 300 _traj.ra traj.ra					;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/signal -F -I -r0.005 -f8 -n300 -1 1.25:1.25:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra signal.ra k_space.ra				;\
	$(TOOLDIR)/index 5 300 tmp1.ra							;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra						;\
	$(TOOLDIR)/moba -P5000 -L -i20 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/moba --irflash P --seq.tr 0.005 -i20 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -t traj.ra k_space.ra TI.ra reco2.ra sens.ra	;\
	$(TOOLDIR)/nrmse -t 0.0005 reco2.ra reco.ra			  		;\
	$(TOOLDIR)/slice 6 1 reco.ra r1map.ra						;\
	$(TOOLDIR)/phantom -x12 -c circ.ra						;\
	$(TOOLDIR)/fmac r1map.ra circ.ra r1masked.ra	    				;\
	$(TOOLDIR)/scale -- 0.8 circ.ra r1ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.002 r1masked.ra r1ref.ra			    		;\
	$(TOOLDIR)/slice 6 2 reco.ra famap.ra						;\
	$(TOOLDIR)/fmac famap.ra circ.ra famasked.ra	    				;\
	$(TOOLDIR)/scale -- 8 circ.ra faref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.005 famasked.ra faref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-t1-alpha-in-non-cartesian: traj repmat phantom signal fmac ones index scale moba slice spow nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x12 -y12 _traj.ra                  ;\
	$(TOOLDIR)/repmat 5 300 _traj.ra traj.ra					;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/signal -F -I -r0.005 -n300 -f8. -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra signal.ra k_space.ra				;\
	$(TOOLDIR)/ones 2 12 12 _famap.ra				;\
	$(TOOLDIR)/scale 8. _famap.ra famap.ra						;\
	$(TOOLDIR)/index 5 300 tmp1.ra							;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra						;\
	$(TOOLDIR)/moba -A famap.ra -L -i11 --spokes-per-TR 12 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/moba --irflash A -A famap.ra -i11 --spokes-per-TR 12 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -t traj.ra k_space.ra TI.ra reco2.ra sens2.ra	;\
	$(TOOLDIR)/nrmse -t 0.0005 reco2.ra reco.ra			  		;\
	$(TOOLDIR)/slice 6 1 reco.ra r1map.ra						;\
	$(TOOLDIR)/spow -- -1. r1map.ra t1map.ra						;\
	$(TOOLDIR)/phantom -x12 -c circ.ra						;\
	$(TOOLDIR)/fmac t1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.0005 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-t1-alpha-in-comparison: traj repmat phantom signal fmac ones index scale moba slice spow nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x12 -y12 _traj.ra                  ;\
	$(TOOLDIR)/repmat 5 300 _traj.ra traj.ra					;\
	$(TOOLDIR)/phantom -k -c -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/signal -F -I -r0.005 -n300 -f8. -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra signal.ra k_space.ra				;\
	$(TOOLDIR)/ones 2 12 12 _famap.ra				;\
	$(TOOLDIR)/scale 8. _famap.ra famap.ra						;\
	$(TOOLDIR)/index 5 300 tmp1.ra							;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra						;\
	$(TOOLDIR)/moba -A famap.ra -L -i11 --spokes-per-TR 12 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -B 0.0001 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/moba --sim.seq F --sim.type O --seq.tr 0.005 --seq.te 0.003 --seq.fa 8 --seq.rf-duration 0.0001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0 -i11 --spokes-per-TR 12 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -B 0.0001 -t traj.ra k_space.ra TI.ra reco2.ra sens2.ra	;\
	$(TOOLDIR)/slice 6 1 reco.ra r1map.ra						;\
	$(TOOLDIR)/slice 6 0 reco2.ra r1map2.ra						;\
	$(TOOLDIR)/nrmse -t 0.007 r1map2.ra r1map.ra			  		;\
	$(TOOLDIR)/spow -- -1. r1map.ra t1map.ra						;\
	$(TOOLDIR)/phantom -x12 -c circ.ra						;\
	$(TOOLDIR)/fmac t1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.0005 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-irflash: traj repmat phantom signal fmac ones scale index moba slice spow looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
	$(TOOLDIR)/phantom -c -k -t traj.ra circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n1000 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra k_space.ra					;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --sim.seq F --sim.type O --seq.tr 0.005 --seq.te 0.003 --seq.fa 8 --seq.rf-duration 0.0001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0 -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -B 0.0001 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra						;\
	$(TOOLDIR)/spow -- -1. r1map.ra t1map.ra						;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac t1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.007 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-b1-input: traj repmat phantom signal fmac ones scale index moba slice spow looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
	$(TOOLDIR)/phantom -c -k -t traj.ra circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n1000 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/ones 2 16 16 _b1map.ra				;\
	$(TOOLDIR)/scale 8 _b1map.ra b1map.ra				;\
	$(TOOLDIR)/fmac circ.ra signal.ra k_space.ra					;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --sim.seq F --sim.type O --seq.tr 0.005 --seq.te 0.003 --seq.fa 8 --seq.rf-duration 0.0001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0 -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -B 0.0001 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/moba --sim.seq F --sim.type O --seq.tr 0.005 --seq.te 0.003 --seq.fa 1 --seq.rf-duration 0.0001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0 --sim.b1map b1map.ra -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -B 0.0001 -t traj.ra k_space.ra TI.ra reco2.ra sens2.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra						;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac r1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/slice 6 0 reco2.ra r1map2.ra						;\
	$(TOOLDIR)/fmac r1map2.ra circ.ra masked2.ra	    				;\
	$(TOOLDIR)/nrmse -t 0.0003 masked.ra masked2.ra			  		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-irflash-multi-coil: traj repmat phantom signal fmac ones scale index moba slice spow looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
	$(TOOLDIR)/phantom -s 3 -c -k -t traj.ra circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n1000 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra k_space.ra					;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --sim.seq F --sim.type O --seq.tr 0.005 --seq.te 0.003 --seq.fa 8 --seq.rf-duration 0.0001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0 -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -B 0.0001 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra r1map.ra						;\
	$(TOOLDIR)/spow -- -1. r1map.ra t1map.ra						;\
	$(TOOLDIR)/phantom -x16 -c circ.ra						;\
	$(TOOLDIR)/fmac t1map.ra circ.ra masked.ra	    				;\
	$(TOOLDIR)/scale -- 1.12 circ.ra ref.ra			    			;\
	$(TOOLDIR)/nrmse -t 0.007 masked.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-moba-bloch-irflash-r2fix: traj repmat phantom signal fmac ones scale index moba slice spow looklocker fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	               		 	;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
	$(TOOLDIR)/phantom -c -k -t traj.ra circ.ra 		                  		;\
	$(TOOLDIR)/signal -I -F -r0.005 -f8 -n1000 -1 1.12:1.12:1 -2 100:100:1 signal.ra	;\
	$(TOOLDIR)/fmac circ.ra signal.ra k_space.ra					;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   						;\
	$(TOOLDIR)/scale 0.005 tmp1.ra TI.ra                    	       		;\
	$(TOOLDIR)/moba --sim.seq F --sim.type O --seq.tr 0.005 --seq.te 0.003 --seq.fa 8 --seq.rf-duration 0.0001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0 -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.001 -B 0.0001 -t traj.ra k_space.ra TI.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 2 reco.ra r2map.ra						;\
	$(TOOLDIR)/ones 2 16 16 ones.ra						;\
	$(TOOLDIR)/nrmse -t 0.00001 ones.ra r2map.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
# tests/test-moba-bloch-irbssfp: traj repmat phantom signal fmac index moba spow slice scale nrmse
# 	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
# 	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
# 	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
# 	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
# 	$(TOOLDIR)/signal -B -I -r 0.0045 -e 0.00225 -f45 -n 1000 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
# 	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
# 	$(TOOLDIR)/index 5 1000 dummy_ti.ra 	;\
# 	$(TOOLDIR)/moba --sim.seq B --sim.type O --seq.tr 0.0045 --seq.te 0.00225 --seq.fa 45 --seq.rf-duration 0.00001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0.0001 --seq.scaling 1:1:1:0:1 -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.0001 -B 0.0001 -t traj.ra k_space.ra dummy_ti.ra reco.ra sens.ra	;\
# 	$(TOOLDIR)/spow -- -1 reco.ra reco2.ra				;\
# 	$(TOOLDIR)/slice 6 0 reco2.ra t1map.ra				;\
# 	$(TOOLDIR)/phantom -x 16 -c ref_geom.ra				;\
# 	$(TOOLDIR)/fmac t1map.ra ref_geom.ra masked_t1.ra				;\
# 	$(TOOLDIR)/scale -- 1.12 ref_geom.ra ref_t1.ra				;\
# 	$(TOOLDIR)/nrmse -t 0.007 masked_t1.ra ref_t1.ra				;\
# 	$(TOOLDIR)/slice 6 2 reco2.ra t2map.ra				;\
# 	$(TOOLDIR)/fmac t2map.ra ref_geom.ra masked_t2.ra				;\
# 	$(TOOLDIR)/scale -- 0.1 ref_geom.ra ref_t2.ra				;\
# 	$(TOOLDIR)/nrmse -t 0.006 masked_t2.ra ref_t2.ra				;\
# 	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
# 	touch $@

# tests/test-moba-bloch-irbssfp-modbloch: traj repmat phantom signal fmac index moba spow modbloch slice nrmse
# 	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
# 	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
# 	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
# 	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
# 	$(TOOLDIR)/signal -B -I -r 0.0045 -e 0.00225 -f45 -n 1000 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
# 	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
# 	$(TOOLDIR)/index 5 1000 dummy_ti.ra 	;\
# 	$(TOOLDIR)/moba --sim.seq B --sim.type O --seq.tr 0.0045 --seq.te 0.00225 --seq.fa 45 --seq.rf-duration 0.00001 --seq.bwtp 4 --seq.inv-pulse-length 0 --seq.prep-pulse-length 0.00001 --seq.scaling 1:1:1:0:1 -i11 -C250 -s0.95 -f1 -R3 -o1 -j0.0001 -B 0.0001 -t traj.ra k_space.ra dummy_ti.ra rreco.ra sens.ra	;\
# 	$(TOOLDIR)/spow -- -1 rreco.ra reco.ra		;\
# 	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001:4 -R2 -f1 -o0 -l1 -n0 -i15 -a1 -m0 -r0 -t traj.ra k_space.ra reco2.ra sens2.ra	;\
# 	$(TOOLDIR)/phantom -x 16 -c ref_geom.ra				;\
# 	$(TOOLDIR)/slice 6 0 reco.ra t1map.ra				;\
# 	$(TOOLDIR)/slice 6 0 reco2.ra t1map2.ra				;\
# 	$(TOOLDIR)/fmac t1map.ra ref_geom.ra masked_t1.ra				;\
# 	$(TOOLDIR)/fmac t1map2.ra ref_geom.ra masked2_t1.ra				;\
# 	$(TOOLDIR)/nrmse -t 0.001 masked_t1.ra masked2_t1.ra				;\
# 	$(TOOLDIR)/slice 6 2 reco.ra t2map.ra				;\
# 	$(TOOLDIR)/slice 6 2 reco2.ra t2map2.ra				;\
# 	$(TOOLDIR)/fmac t2map.ra ref_geom.ra masked_t2.ra				;\
# 	$(TOOLDIR)/fmac t2map2.ra ref_geom.ra masked2_t2.ra				;\
# 	$(TOOLDIR)/nrmse -t 0.004 masked_t2.ra masked2_t2.ra				;\
# 	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
# 	touch $@

TESTS_SLOW += tests/test-moba-t1 tests/test-moba-t1-sms tests/test-moba-t1-no-IR
TESTS_SLOW += tests/test-moba-t1-magn tests/test-moba-t1-nonCartesian tests/test-moba-t1-nufft
TESTS_SLOW += tests/test-moba-t1-MOLLI
TESTS_SLOW += tests/test-moba-t1-IR_SS
TESTS_SLOW += tests/test-moba-t1-irgnm-admm
TESTS_SLOW += tests/test-moba-t2
TESTS_SLOW += tests/test-moba-meco-noncart-r2s tests/test-moba-meco-noncart-wfr2s
TESTS_SLOW += tests/test-moba-t1-alpha-non-cartesian
TESTS_SLOW += tests/test-moba-t1-alpha-in-non-cartesian
TESTS_SLOW += tests/test-moba-bloch-irflash tests/test-moba-bloch-t1-alpha-in-comparison
TESTS_SLOW += tests/test-moba-bloch-b1-input tests/test-moba-bloch-irflash-multi-coil tests/test-moba-bloch-irflash-r2fix
