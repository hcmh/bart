
tests/test-mdb-bloch: phantom sim fmac fft ones modbloch slice scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x16 -c basis_geom.ra				;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000:0 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra image.ra		;\
	$(TOOLDIR)/fft 3 image.ra k_space.ra					;\
	$(TOOLDIR)/ones 6 16 16 1 1 1 1000 psf.ra		;\
	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001 -R2 -f1 -o0 -l1 -n0 -i15 -a1 -m0 -r0 -p psf.ra k_space.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra t1map.ra				;\
	$(TOOLDIR)/fmac t1map.ra basis_geom.ra masked_t1.ra				;\
	$(TOOLDIR)/scale -- 1.12 basis_geom.ra ref_t1.ra				;\
	$(TOOLDIR)/nrmse -t 0.007 masked_t1.ra ref_t1.ra				;\
	$(TOOLDIR)/slice 6 2 reco.ra t2map.ra				;\
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
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000:0 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
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
	$(TOOLDIR)/slice 6 2 reco.ra t2map.ra				;\
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
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000:0 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001 -R2 -f1 -o0 -l1 -n0 -i15 -a1 -m0 -r0 -t traj.ra k_space.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 0 reco.ra t1map.ra				;\
	$(TOOLDIR)/phantom -x 16 -c ref_geom.ra				;\
	$(TOOLDIR)/fmac t1map.ra ref_geom.ra masked_t1.ra				;\
	$(TOOLDIR)/scale -- 1.12 ref_geom.ra ref_t1.ra				;\
	$(TOOLDIR)/nrmse -t 0.007 masked_t1.ra ref_t1.ra				;\
	$(TOOLDIR)/slice 6 2 reco.ra t2map.ra				;\
	$(TOOLDIR)/fmac t2map.ra ref_geom.ra masked_t2.ra				;\
	$(TOOLDIR)/scale -- 0.1 ref_geom.ra ref_t2.ra				;\
	$(TOOLDIR)/nrmse -t 0.005 masked_t2.ra ref_t2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-mdb-bloch tests/test-mdb-bloch-psf tests/test-mdb-bloch-traj



