T_TOL=1e-5

TRAJ_TURNS=/home/ague/data/traj_tests/t_turns.cfl
O_TRAJ_TURNS=-x128 -y73 -t5 -D

TRAJ_GA_c=/home/ague/data/traj_tests/t_GA_c.cfl
O_TRAJ_GA_c=-x128 -y51 -r -G -c

TRAJ_GA_H=/home/ague/data/traj_tests/t_GA_H.cfl
O_TRAJ_GA_H=-x128 -y53 -r -H

TRAJ_tiny_GA=/home/ague/data/traj_tests/t_tiny_GA.cfl
O_TRAJ_tiny_GA=-x128 -y127 -s11 -G -t10 -D -r

TRAJ_MEMS=/home/ague/data/traj_tests/t_MEMS.cfl
O_TRAJ_MEMS=-x128 -y31 -t7 -r -s3 -D -E -e5 -c
 
TRAJ_MEMS_ASYM=/home/ague/data/traj_tests/t_MEMS_asym.cfl
O_TRAJ_MEMS_ASYM=-x128 -d192 -y31 -t7 -r -s3 -D -E -e5 -c
 
tests/test-traj_turns: traj nrmse ${TRAJ_TURNS}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_TURNS} t_turns.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_turns.ra $(basename ${TRAJ_TURNS} .cfl)		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_GA_c: traj nrmse ${TRAJ_GA_c}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_GA_c} t_GA_c.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_GA_c.ra $(basename ${TRAJ_GA_c} .cfl)		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
tests/test-traj_GA_H: traj nrmse ${TRAJ_GA_H}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_GA_H} t_GA_H.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_GA_H.ra $(basename ${TRAJ_GA_H} .cfl)		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
tests/test-traj_tiny_GA: traj nrmse ${TRAJ_tiny_GA}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_tiny_GA} t_tiny_GA.ra				;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_tiny_GA.ra $(basename ${TRAJ_tiny_GA} .cfl)	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
tests/test-traj_MEMS: traj nrmse ${TRAJ_MEMS}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_MEMS} t_MEMS.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_MEMS.ra $(basename ${TRAJ_MEMS} .cfl)		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
tests/test-traj_MEMS_ASYM: traj nrmse ${TRAJ_MEMS_ASYM}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_MEMS_ASYM} t_MEMS_asym.ra				;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_MEMS_asym.ra $(basename ${TRAJ_MEMS_ASYM} .cfl)		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

TESTS_AGUE += tests/test-traj_turns tests/test-traj_GA_c tests/test-traj_GA_H tests/test-traj_tiny_GA tests/test-traj_MEMS
TESTS_AGUE += tests/test-traj_MEMS_ASYM


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
