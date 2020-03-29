
tests/test-sim-analy-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000 -1 3:3:1 -2 1:1:1 analytical.ra	;\
	$(TOOLDIR)/sim -P 0:1:0.0045:0.00225:0.00001:45:1000 -1 3:3:1 -2 1:1:1 simu.ra		;\
	$(TOOLDIR)/nrmse -t 0.008 analytical.ra simu.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-analy-flash: sim cabs mip spow fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:2:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 analytical.ra		;\
	$(TOOLDIR)/sim -P 0:2:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 _simu.ra			;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra simu.ra							;\
	$(TOOLDIR)/nrmse -t 0.002 analytical.ra simu.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-analy-irflash: sim cabs mip spow fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:5:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 analytical.ra		;\
	$(TOOLDIR)/sim -P 0:5:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 _simu.ra			;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra simu.ra							;\
	$(TOOLDIR)/nrmse -t 0.005 analytical.ra simu.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:0:0.0045:0.00225:0.001:45:1000 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:0:0.0045:0.00225:0.001:45:1000 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.005 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:1:0.0045:0.00225:0.001:45:1000 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:1:0.0045:0.00225:0.001:45:1000 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.003 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:2:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:2:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.002 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:5:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:5:0.0041:0.00258:0.001:6:1000 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.001 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-sim-analy-irbssfp tests/test-sim-analy-flash tests/test-sim-analy-irflash tests/test-sim-ode-bssfp tests/test-sim-ode-irbssfp tests/test-sim-ode-flash tests/test-sim-ode-irflash