
tests/test-estdelay: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/traj -D -r -y8 n.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay n.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.00001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-c: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 -c t.ra						;\
	$(TOOLDIR)/traj -D -r -y8 -c n.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay n.ra k.ra` -r -y8 -c t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@	

tests/test-estdelay-transverse: estdelay traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -O -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/traj -D -r -y8 n.ra						;\
	$(TOOLDIR)/phantom -k -t t.ra k.ra							;\
	$(TOOLDIR)/traj -D -O -q`$(TOOLDIR)/estdelay n.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.0011 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-coils: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -G -q0.3:-0.1:0.2 -y5 t.ra						;\
	$(TOOLDIR)/traj -G -r -y5 n.ra								;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -s8 -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -G -q`$(TOOLDIR)/estdelay ns.ra k.ra` -y5 t2.ra				;\
	$(TOOLDIR)/nrmse -t 0.004 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-coils-c: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -G -q0.3:-0.1:0.2 -y5 -c t.ra						;\
	$(TOOLDIR)/traj -G -r -y5 -c n.ra								;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -s8 -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -G -q`$(TOOLDIR)/estdelay ns.ra k.ra` -c -y5 t2.ra				;\
	$(TOOLDIR)/nrmse -t 0.004 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-ring: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -O -y5 t.ra					;\
	$(TOOLDIR)/traj -D -r -y5 n.ra							;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -R ns.ra k.ra` -O -y5 t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-ring-c: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -O -c -y5 t.ra					;\
	$(TOOLDIR)/traj -D -c -r -y5 n.ra							;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -R ns.ra k.ra` -O -c -y5 t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@	

tests/test-estdelay-ring-coils: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -O -y5 t.ra					;\
	$(TOOLDIR)/traj -D -r -y5 n.ra							;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -k -s8 -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -R ns.ra k.ra` -O -y5 t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@	

tests/test-estdelay-ring-coils-c: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -O -c -y5 t.ra					;\
	$(TOOLDIR)/traj -D -c -r -y5 n.ra							;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -k -s8 -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -R ns.ra k.ra` -O -c -y5 t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-ring-b0: estdelay scale index zexp fmac traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -r -q0.3:0.1:0.2 -O -c -y5 t.ra					;\
	$(TOOLDIR)/traj -D -c -r -y5 n.ra							;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/phantom -k -s8 -t ts.ra k.ra							;\
	$(TOOLDIR)/index 2 5 i.ra								;\
	$(TOOLDIR)/zexp -i i.ra ii.ra								;\
	$(TOOLDIR)/fmac k.ra ii.ra ki.ra							;\
	$(TOOLDIR)/traj -D -r -q`$(TOOLDIR)/estdelay -b -R ns.ra ki.ra` -O -c -y5 t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00015 t.ra t2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-estdelay-scale: estdelay scale traj phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -D -q1.5:1:-0.5 -r -y8 t.ra						;\
	$(TOOLDIR)/scale 0.5 t.ra ts.ra								;\
	$(TOOLDIR)/traj -D -r -y8 n.ra						;\
	$(TOOLDIR)/scale 0.5 n.ra ns.ra								;\
	$(TOOLDIR)/phantom -k -t ts.ra k.ra							;\
	$(TOOLDIR)/traj -D -q`$(TOOLDIR)/estdelay ns.ra k.ra` -r -y8 t2.ra			;\
	$(TOOLDIR)/scale 0.5 t2.ra t2s.ra							;\
	$(TOOLDIR)/nrmse -t 0.0001 ts.ra t2s.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-estdelay tests/test-estdelay-c tests/test-estdelay-transverse
TESTS += tests/test-estdelay-ring tests/test-estdelay-ring-c tests/test-estdelay-coils tests/test-estdelay-coils-c tests/test-estdelay-scale
TESTS += tests/test-estdelay-ring-coils tests/test-estdelay-ring-coils-c tests/test-estdelay-ring-b0

