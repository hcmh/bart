
tests/test-mdb-bloch: phantom sim fmac fft ones modbloch slice scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x16 -c basis_geom.ra				;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra image.ra		;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra					;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 1000 psf.ra		;\
	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001 -R2 -f1 -o0 -l1 -n0 -i15 -a1 -m0 -r0 -p psf.ra k_space.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra t1map.ra				;\
	$(TOOLDIR)/fmac t1map.ra basis_geom.ra masked_t1.ra				;\
	$(TOOLDIR)/scale -- 1.12 basis_geom.ra ref_t1.ra				;\
	$(TOOLDIR)/nrmse -t 0.007 masked_t1.ra ref_t1.ra				;\
	$(TOOLDIR)/slice 6 1 reco.ra t2map.ra				;\
	$(TOOLDIR)/fmac t2map.ra basis_geom.ra masked_t2.ra				;\
	$(TOOLDIR)/scale -- 0.1 basis_geom.ra ref_t2.ra				;\
	$(TOOLDIR)/nrmse -t 0.005 masked_t2.ra ref_t2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mdb-bloch-psf: traj repmat phantom sim fmac ones nufft fft modbloch slice scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/ones 6 1 16 16 1 1 1000 ones.ra		;\
	$(TOOLDIR)/nufft -d 16:16:1 -a traj.ra ones.ra psf.ra	;\
	$(TOOLDIR)/fft -u 3 psf.ra pattern.ra	;\
	$(TOOLDIR)/nufft -d 16:16:1 -a traj.ra k_space.ra zerofill.ra		;\
	$(TOOLDIR)/fft -u 3 zerofill.ra k_grid.ra	;\
	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001 -R2 -f1 -o0 -l1 -n0 -i15 -a1 -m0 -r0 -p pattern.ra k_grid.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra t1map.ra				;\
	$(TOOLDIR)/phantom -x 16 -c ref_geom.ra				;\
	$(TOOLDIR)/fmac t1map.ra ref_geom.ra masked_t1.ra				;\
	$(TOOLDIR)/scale -- 1.12 ref_geom.ra ref_t1.ra				;\
	$(TOOLDIR)/nrmse -t 0.007 masked_t1.ra ref_t1.ra				;\
	$(TOOLDIR)/slice 6 1 reco.ra t2map.ra				;\
	$(TOOLDIR)/fmac t2map.ra ref_geom.ra masked_t2.ra				;\
	$(TOOLDIR)/scale -- 0.1 ref_geom.ra ref_t2.ra				;\
	$(TOOLDIR)/nrmse -t 0.005 masked_t2.ra ref_t2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mdb-bloch-traj: traj repmat phantom sim fmac modbloch slice scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001 -R2 -f1 -o0 -l1 -n0 -i15 -a1 -m0 -r0 -t traj.ra k_space.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra t1map.ra				;\
	$(TOOLDIR)/phantom -x 16 -c ref_geom.ra				;\
	$(TOOLDIR)/fmac t1map.ra ref_geom.ra masked_t1.ra				;\
	$(TOOLDIR)/scale -- 1.12 ref_geom.ra ref_t1.ra				;\
	$(TOOLDIR)/nrmse -t 0.007 masked_t1.ra ref_t1.ra				;\
	$(TOOLDIR)/slice 6 1 reco.ra t2map.ra				;\
	$(TOOLDIR)/fmac t2map.ra ref_geom.ra masked_t2.ra				;\
	$(TOOLDIR)/scale -- 0.1 ref_geom.ra ref_t2.ra				;\
	$(TOOLDIR)/nrmse -t 0.005 masked_t2.ra ref_t2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mdb-t1: phantom sim fmac fft ones index scale moba looklocker transpose slice mip scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                	;\
	$(TOOLDIR)/phantom -x 32 -T -b basis_geom.ra                    	;\
	$(TOOLDIR)/sim -n 10 -P 1:5:0.0045:0.00225:0.00001:6:1000 basis_simu.ra  ;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra basis_simu.ra image.ra		;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra					;\
	$(TOOLDIR)/ones 16 32 32 1 1 1 1000 1 1 1 1 1 1 1 1 1 1 psf.ra		;\
	$(TOOLDIR)/index 5 1000 tmp1.ra   					;\
	$(TOOLDIR)/scale 0.0045 tmp1.ra TI.ra                           	;\
	$(TOOLDIR)/moba -L -l1 -i10 -j0.001 -f1 -C300 -s0.95 -B0.3 -p psf.ra k_space.ra TI.ra reco.ra	;\
	$(TOOLDIR)/looklocker -t0.1 -D0.0 -R4.5e-3 reco.ra T1.ra    		;\
	$(TOOLDIR)/transpose 6 7 basis_geom.ra mask.ra	            		;\
	$(TOOLDIR)/fmac T1.ra mask.ra segments.ra		    		;\
	$(TOOLDIR)/slice 6 0 7 1 segments.ra tube.ra	            		;\
	$(TOOLDIR)/mip 7 tube.ra mean.ra			    		;\
	$(TOOLDIR)/ones 1 1 ones.ra				    		;\
	$(TOOLDIR)/scale -- 0.877 ones.ra ref.ra		    		;\
	$(TOOLDIR)/nrmse -t 0.01 mean.ra ref.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mdb-t1-traj: traj phantom sim fmac nufft fft ones index scale moba looklocker resize transpose slice mip scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x32 -y1 -r -D -G -s7 -t1000 _traj.ra   		;\
	$(TOOLDIR)/transpose 5 10 _traj.ra _traj2.ra   				;\
	$(TOOLDIR)/scale 0.5 _traj2.ra traj.ra  				;\
	$(TOOLDIR)/phantom -T -k -b -t traj.ra basis_geom.ra  			;\
	$(TOOLDIR)/sim -n 10 -P 1:5:0.0045:0.00225:0.00001:6:1000 basis_simu.ra  ;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra basis_simu.ra data.ra	   	;\
	$(TOOLDIR)/ones 16 1 32 1 1 1 1000 1 1 1 1 1 1 1 1 1 1 ones.ra	   	;\
	$(TOOLDIR)/nufft -d 32:32:1 -a _traj2.ra ones.ra pattern.ra	   	;\
	$(TOOLDIR)/fft -u 3 pattern.ra psf.ra				   	;\
	$(TOOLDIR)/nufft -d 32:32:1 -a _traj2.ra data.ra zerofill_reco.ra  	;\
	$(TOOLDIR)/fft -u 3 zerofill_reco.ra k_space.ra			   	;\
	$(TOOLDIR)/index 5 1000 tmp1.ra    				   	;\
	$(TOOLDIR)/scale 0.0045 tmp1.ra TI.ra   			   	;\
 	$(TOOLDIR)/moba -L -l1 -i7 -j0.01 -C300 -s0.95 -B0.3 -p psf.ra k_space.ra TI.ra reco.ra  ;\
	$(TOOLDIR)/looklocker -t0.1 -D0.0 -R4.5e-3 reco.ra T1.ra    		;\
	$(TOOLDIR)/resize -c 0 16 1 16 T1.ra T1_crop.ra             		;\
	$(TOOLDIR)/phantom -x 16 -T -b basis_geom_px.ra             		;\
	$(TOOLDIR)/transpose 6 7 basis_geom_px.ra mask.ra	    		;\
	$(TOOLDIR)/fmac T1_crop.ra mask.ra segments.ra		    		;\
	$(TOOLDIR)/slice 6 0 7 1 segments.ra tube.ra	            		;\
	$(TOOLDIR)/mip 7 tube.ra mean.ra			    		;\
	$(TOOLDIR)/ones 1 1 ones.ra				    		;\
	$(TOOLDIR)/scale -- 0.877 ones.ra ref.ra		    		;\
	$(TOOLDIR)/nrmse -t 0.02 mean.ra ref.ra	                    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-mdb-bloch tests/test-mdb-bloch-psf tests/test-mdb-bloch-traj

# fail on radon for some unknown reason
# TESTS += tests/test-mdb-t1 tests/test-mdb-t1-traj



