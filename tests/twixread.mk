
TWIXFILE1=/home/ague/archive/2019/2019-03-25_Other-MU_Export/single_raid/meas_MID00311_FID65562_t1_fl2d.dat
TWIXFILE2=/home/ague/data/T1825/T1825.dat
TWIXFILE3=/home/ague/archive/2018/2018-04-13_RT-Export/U00000013.dat

tests/test-twixread: twixread ${TWIXFILE1}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIXFILE1} ksp.ra					;\
	echo "c304bf3bb41c7571408e776ec1280870 *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-mpi: twixread ${TWIXFILE2}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -M -x320 -r11 -s3 -n217 -c34 ${TWIXFILE2} ksp.ra		;\
	echo "9ee480b68ed70a7388851277e209b38d *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-umgread: umgread ${TWIXFILE3}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/umgread -x320 -y35 -c20 -p160 ${TWIXFILE3} ksp.ra			;\
	echo "e857ec5cc919398f5f6f88d36c27adfa *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS_AGUE += tests/test-twixread tests/test-twixread-mpi tests/test-umgread

