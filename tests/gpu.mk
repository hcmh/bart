

# 

tests/test-pics-gpu: pics nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/pics -g -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco1.ra	;\
	$(TOOLDIR)/pics    -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-gpu-noncart: traj scale phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics    -S -r0.001 -t traj2.ra ksp.ra o.ra reco1.ra			;\
	$(TOOLDIR)/pics -g -S -r0.001 -t traj2.ra ksp.ra o.ra reco2.ra			;\
	$(TOOLDIR)/nrmse -t 0.001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-gpu-weights: pics scale ones nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 2 128 128 weights.ra						;\
	$(TOOLDIR)/scale 0.1 weights.ra weights2.ra					;\
	$(TOOLDIR)/pics    -S -r0.001 -p weights2.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco1.ra	;\
	$(TOOLDIR)/pics -g -S -r0.001 -p weights2.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 reco2.ra reco1.ra				 	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# similar to the non-gpu test this had to be relaxed to 0.01
tests/test-pics-gpu-noncart-weights: traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y32 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra					;\
	$(TOOLDIR)/ones 4 1 256 32 1 weights.ra						;\
	$(TOOLDIR)/scale 0.1 weights.ra weights2.ra					;\
	$(TOOLDIR)/pics    -S -r0.001 -p weights2.ra -t traj2.ra ksp.ra $(TESTS_OUT)/coils.ra reco1.ra ;\
	$(TOOLDIR)/pics -g -S -r0.001 -p weights2.ra -t traj2.ra ksp.ra $(TESTS_OUT)/coils.ra reco2.ra ;\
	$(TOOLDIR)/nrmse -t 0.010 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-mdb-gpu-bloch-traj: traj repmat phantom sim fmac modbloch nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x16 -y16 _traj.ra			;\
	$(TOOLDIR)/repmat 5 1000 _traj.ra traj.ra	;\
	$(TOOLDIR)/phantom -c -k -t traj.ra basis_geom.ra				;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000:0:0.00001:0.00001:4 -1 1.12:1.12:1 -2 0.1:0.1:1 basis_simu.ra	;\
	$(TOOLDIR)/fmac basis_geom.ra basis_simu.ra k_space.ra		;\
	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001:4 -R2 -f1 -o0 -l1 -n0 -i15 -a1 -m0 -r0 -t traj.ra k_space.ra reco1.ra sens1.ra	;\
	$(TOOLDIR)/modbloch -P 1:0.0045:0.00225:45:0.00001:0.00001:0.00001:4 -R2 -f1 -o0 -g -l1 -n0 -i15 -a1 -m0 -r0 -t traj.ra k_space.ra reco2.ra sens2.ra	;\
	$(TOOLDIR)/nrmse -t 0.001 reco1.ra reco2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# test using as many of the first 16 GPUs as possible
tests/test-pics-multigpu: pics repmat nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 5 32 $(TESTS_OUT)/shepplogan_coil_ksp.ra kspaces.ra		;\
	$(TOOLDIR)/pics -g -G65535 -r0.01 -L32 kspaces.ra $(TESTS_OUT)/coils.ra reco1.ra		;\
	$(TOOLDIR)/pics -g         -r0.01 -L32 kspaces.ra $(TESTS_OUT)/coils.ra reco2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS_GPU += tests/test-pics-gpu tests/test-pics-gpu-noncart
TESTS_GPU += tests/test-pics-gpu-weights tests/test-pics-gpu-noncart-weights
TESTS_GPU += tests/test-mdb-gpu-bloch-traj
TESTS_GPU += tests/test-pics-multigpu
