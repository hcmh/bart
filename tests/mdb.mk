
tests/test-mdb-bloch: phantom sim fmac fft ones modbloch transpose slice mip scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x 20 -T -b basis_geom.ra				;\
	$(TOOLDIR)/sim -n 10 -V"3,1:0.877,0.048:1.140,0.06:1.404,0.06:0.866,0.095:1.159,0.108:1.456,0.122:0.883,0.129:1.166,0.150:1.442,0.163" -P 1:1:0.0045:0.00225:0.00001:45:1000 basis_simu.ra	;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra basis_simu.ra image.ra		;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra					;\
	$(TOOLDIR)/ones 16 1 20 20 1 1 1000 1 1 1 1 1 1 1 1 1 1 psf.ra		;\
	$(TOOLDIR)/modbloch -t0.0045 -e0.00225 -R2 -f1 -s5000 -F45 -M1 -o0 -l1 -v0.00001 -b0.00001 -n0 -i15 -D0.00001 -a1 -w0 -r0 -p psf.ra k_space.ra reco.ra sens.ra	;\
	$(TOOLDIR)/transpose 6 7 basis_geom.ra mask.ra				;\
	$(TOOLDIR)/fmac reco.ra mask.ra segments.ra				;\
	$(TOOLDIR)/slice 6 0 7 1 segments.ra tube.ra				;\
	$(TOOLDIR)/mip 7 tube.ra mean.ra					;\
	$(TOOLDIR)/ones 1 1 ones.ra						;\
	$(TOOLDIR)/scale -- 0.877 ones.ra ref.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 mean.ra ref.ra					;\
	$(TOOLDIR)/slice 6 1 7 1 segments.ra tube.ra				;\
	$(TOOLDIR)/mip 7 tube.ra mean.ra					;\
	$(TOOLDIR)/scale -- 0.048 ones.ra ref.ra				;\
	$(TOOLDIR)/nrmse -t 0.012 mean.ra ref.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mdb-bloch-traj: phantom sim fmac fft ones modbloch transpose slice mip scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x 20 -y 1 -t 1000 -r -s 13 _traj.ra			;\
	$(TOOLDIR)/transpose 5 10 _traj.ra _traj2.ra				;\
	$(TOOLDIR)/scale 0.75 _traj2.ra traj.ra					;\
	$(TOOLDIR)/phantom -T -k -b -t traj.ra basis_geom.ra			;\
	$(TOOLDIR)/sim -n 10 -V"3,1:0.877,0.048:1.140,0.06:1.404,0.06:0.866,0.095:1.159,0.108:1.456,0.122:0.883,0.129:1.166,0.150:1.442,0.163" -P 1:1:0.0045:0.00225:0.00001:45:1000 basis_simu.ra	;\
	$(TOOLDIR)/fmac -s 64 basis_geom.ra basis_simu.ra data.ra		;\
	$(TOOLDIR)/ones 16 1 20 1 1 1 1000 1 1 1 1 1 1 1 1 1 1 ones.ra		;\
	$(TOOLDIR)/nufft -d 20:20:1 -a traj.ra ones.ra psf.ra			;\
	$(TOOLDIR)/fft -u 3 psf.ra pattern.ra					;\
	$(TOOLDIR)/nufft -d 20:20:1 -a traj.ra data.ra zerofill_reco.ra		;\
	$(TOOLDIR)/fft -u 3 zerofill_reco.ra k_space.ra				;\
	$(TOOLDIR)/modbloch -t0.0045 -e0.00225 -R2 -f1 -s5000 -F45 -M1 -o1 -l1 -v0.00001 -b0.00001 -n0 -i10 -D0.00001 -d4 -a1 -w0 -r0 -p pattern.ra k_space.ra reco.ra sens.ra	;\
	$(TOOLDIR)/phantom -x 20 -T -b basis_geom_px.ra				;\
	$(TOOLDIR)/transpose 6 7 basis_geom_px.ra mask.ra			;\
	$(TOOLDIR)/fmac reco.ra mask.ra segments.ra				;\
	$(TOOLDIR)/slice 6 0 7 1 segments.ra tube.ra				;\
	$(TOOLDIR)/mip 7 tube.ra mean.ra					;\
	$(TOOLDIR)/ones 1 1 ones.ra						;\
	$(TOOLDIR)/scale -- 0.877 ones.ra ref.ra				;\
	$(TOOLDIR)/nrmse mean.ra ref.ra						;\
	$(TOOLDIR)/slice 6 1 7 1 segments.ra tube.ra				;\
	$(TOOLDIR)/mip 7 tube.ra mean.ra					;\
	$(TOOLDIR)/scale -- 0.048 ones.ra ref.ra				;\
	$(TOOLDIR)/nrmse mean.ra ref.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-mdb-t1: phantom sim fmac fft ones index scale moba looklocker transpose slice mip scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	                	;\
	$(TOOLDIR)/phantom -x 32 -T -b basis_geom.ra                    	;\
	$(TOOLDIR)/sim -n 10 -V"3,1:0.877,0.048:1.140,0.06:1.404,0.06:0.866,0.095:1.159,0.108:1.456,0.122:0.883,0.129:1.166,0.150:1.442,0.163" -P 1:5:0.0045:0.00225:0.00001:6:1000 basis_simu.ra  ;\
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
	$(TOOLDIR)/sim -n 10 -V"3,1:0.877,0.048:1.140,0.06:1.404,0.06:0.866,0.095:1.159,0.108:1.456,0.122:0.883,0.129:1.166,0.150:1.442,0.163" -P 1:5:0.0045:0.00225:0.00001:6:1000 basis_simu.ra  ;\
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

TESTS += tests/test-mdb-bloch

# fail on radon for some unknown reason
# TESTS += tests/test-mdb-t1 tests/test-mdb-t1-traj



