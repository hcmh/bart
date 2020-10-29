
T_TOL=1e-5

TRAJ_TURNS=/home/ague/data/traj_tests/t_turns.cfl
O_TRAJ_TURNS="-x128 -y73 -t5 -D"

TRAJ_GA_c=/home/ague/data/traj_tests/t_GA_c.cfl
O_TRAJ_GA_c="-x128 -y51 -r -G -c"

TRAJ_GA_H=/home/ague/data/traj_tests/t_GA_H.cfl
O_TRAJ_GA_H="-x128 -y53 -r -H"

TRAJ_tiny_GA=/home/ague/data/traj_tests/t_tiny_GA.cfl
O_TRAJ_tiny_GA="-x128 -y127 -s11 -G -t10 -D -r"

TRAJ_MEMS=/home/ague/data/traj_tests/t_MEMS.cfl
O_TRAJ_MEMS="-x128 -y31 -t7 -r -s3 -D -E -e5 -c"
 
TRAJ_MEMS_ASYM=/home/ague/data/traj_tests/t_MEMS_asym.cfl
O_TRAJ_MEMS_ASYM="-x128 -d192 -y31 -t7 -r -s3 -D -E -e5 -c"
 
tests/test-traj_turns: traj nrmse ${TRAJ_TURNS}
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
