tests/test-sim-to-signal-irflash: sim signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:5:0.0041:0.00258:0.001:8:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 sim.ra	;\
	$(TOOLDIR)/signal -I -F -r0.0041 -e0.00258 -f8 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-flash: sim signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:2:0.0041:0.00258:0.001:8:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 sim.ra	;\
	$(TOOLDIR)/signal -F -r0.0041 -e0.00258 -f8 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-to-signal-irbSSFP: sim signal nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 sim.ra	;\
	$(TOOLDIR)/signal -I -B -r0.0045 -e0.00225 -f45 -n1000 -1 3:3:1 -2 1:1:1 signal.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 sim.ra signal.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-analy-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:1:0.0045:0.00225:0.001:45:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 analytical.ra	;\
	$(TOOLDIR)/sim -P 0:1:0.0045:0.00225:0.00001:45:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu.ra		;\
	$(TOOLDIR)/nrmse -t 0.011 analytical.ra simu.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-analy-flash: sim cabs mip spow fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:2:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 analytical.ra		;\
	$(TOOLDIR)/sim -P 0:2:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 _simu.ra			;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra simu.ra							;\
	$(TOOLDIR)/nrmse -t 0.002 analytical.ra simu.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-analy-irflash: sim cabs mip spow fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:5:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 analytical.ra		;\
	$(TOOLDIR)/sim -P 0:5:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 _simu.ra			;\
	$(TOOLDIR)/cabs _simu.ra _simu_abs.ra								;\
	$(TOOLDIR)/mip 32 _simu_abs.ra max.ra								;\
	$(TOOLDIR)/spow -- -1 max.ra scale.ra								;\
	$(TOOLDIR)/fmac _simu.ra scale.ra simu.ra							;\
	$(TOOLDIR)/nrmse -t 0.007 analytical.ra simu.ra			    		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-analy-antihsfp: sim extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 1:4:0.0045:0.00225:0.00001:1:851:0:0.00001:4 -1 0.781:0.781:1 -2 0.065:0.065:1 -o analyt.ra		;\
	$(TOOLDIR)/sim -P 0:4:0.0045:0.00225:0.00001:1:851:0:0.00001:4 -1 0.781:0.781:1 -2 0.065:0.065:1 -o -r rad.ra simu.ra		;\
	$(TOOLDIR)/nrmse -t 0.58 rad.ra analyt.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-bssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:0:0.0045:0.00225:0.001:45:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:0:0.0045:0.00225:0.001:45:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.0075 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-irbssfp: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:1:0.0045:0.00225:0.001:45:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:1:0.0045:0.00225:0.001:45:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.005 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-flash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:2:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:2:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.002 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-sim-ode-irflash: sim nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/sim -P 0:5:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_obs.ra		;\
	$(TOOLDIR)/sim -o -P 0:5:0.0041:0.00258:0.001:6:1000:0:0.00001:4 -1 3:3:1 -2 1:1:1 simu_ode.ra	;\
	$(TOOLDIR)/nrmse -t 0.001 simu_obs.ra simu_ode.ra			    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-sim-to-signal-irflash tests/test-sim-to-signal-flash tests/test-sim-to-signal-irbSSFP
TESTS += tests/test-sim-analy-irbssfp tests/test-sim-analy-flash tests/test-sim-analy-irflash tests/test-sim-analy-antihsfp
TESTS += tests/test-sim-ode-bssfp tests/test-sim-ode-irbssfp tests/test-sim-ode-flash tests/test-sim-ode-irflash
