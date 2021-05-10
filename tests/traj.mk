


tests/test-traj-rot: traj phantom estshift
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -R0. -r -y360 -D t0.ra 				;\
	$(TOOLDIR)/phantom -k -t t0.ra k0.ra 				;\
	$(TOOLDIR)/traj -R30. -r -y360 -D t30.ra			;\
	$(TOOLDIR)/phantom -k -t t30.ra k30.ra 				;\
	$(TOOLDIR)/estshift 4 k0.ra k30.ra | grep "30.00000" 		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




tests/test-traj-zusamp: traj zeros slice join nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y2 -G -s7 -m3 -t2 t.ra 			;\
	$(TOOLDIR)/traj -x128 -y2 -G -s7 -m12 -t2 -z1:4 tz.ra   	;\
	$(TOOLDIR)/slice 2 0 10 0 13 0 t.ra t_0_0_0.ra			;\
	$(TOOLDIR)/slice 2 0 10 0 13 1 t.ra t_0_0_1.ra			;\
	$(TOOLDIR)/slice 2 0 10 0 13 2 t.ra t_0_0_2.ra			;\
	$(TOOLDIR)/slice 2 1 10 0 13 0 t.ra t_1_0_0.ra			;\
	$(TOOLDIR)/slice 2 1 10 0 13 1 t.ra t_1_0_1.ra			;\
	$(TOOLDIR)/slice 2 1 10 0 13 2 t.ra t_1_0_2.ra			;\
	$(TOOLDIR)/slice 2 0 10 1 13 0 t.ra t_0_1_0.ra			;\
	$(TOOLDIR)/slice 2 0 10 1 13 1 t.ra t_0_1_1.ra			;\
	$(TOOLDIR)/slice 2 0 10 1 13 2 t.ra t_0_1_2.ra			;\
	$(TOOLDIR)/slice 2 1 10 1 13 0 t.ra t_1_1_0.ra			;\
	$(TOOLDIR)/slice 2 1 10 1 13 1 t.ra t_1_1_1.ra			;\
	$(TOOLDIR)/slice 2 1 10 1 13 2 t.ra t_1_1_2.ra			;\
	$(TOOLDIR)/zeros 2 3 128 z.ra					;\
	$(TOOLDIR)/join 13 z.ra z.ra z.ra z.ra z.ra t_0_0_0.ra t_0_0_1.ra z.ra t_0_0_2.ra z.ra z.ra z.ra T_0_0.ra ;\
	$(TOOLDIR)/join 13 z.ra z.ra z.ra z.ra t_1_0_0.ra z.ra t_1_0_1.ra z.ra t_1_0_2.ra z.ra z.ra z.ra T_1_0.ra ;\
	$(TOOLDIR)/join 13 z.ra z.ra z.ra z.ra z.ra z.ra t_0_1_0.ra z.ra z.ra t_0_1_1.ra z.ra t_0_1_2.ra T_0_1.ra ;\
	$(TOOLDIR)/join 13 z.ra z.ra z.ra z.ra t_1_1_0.ra z.ra t_1_1_1.ra z.ra z.ra z.ra z.ra t_1_1_2.ra T_1_1.ra ;\
	$(TOOLDIR)/join 2 T_0_0.ra T_1_0.ra T_0.ra 			;\
	$(TOOLDIR)/join 2 T_0_1.ra T_1_1.ra T_1.ra 			;\
	$(TOOLDIR)/join 10 T_0.ra T_1.ra T.ra      			;\
	$(TOOLDIR)/nrmse -t 0.00001 tz.ra T.ra	   			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




# compare customAngle to default angle


tests/test-traj-custom: traj poly nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y128 -r traja.ra				;\
	$(TOOLDIR)/poly 128 1 0 0.0245436926 angle.ra		;\
	$(TOOLDIR)/traj -x128 -y128 -r -C angle.ra trajb.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 traja.ra trajb.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-traj-custom
TESTS += tests/test-traj-zusamp tests/test-traj-rot

