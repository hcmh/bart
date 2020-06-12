
TWIXFILE1=/home/ague/archive/pha/2019/2019-03-25_Other-MU_Export/single_raid/meas_MID00311_FID65562_t1_fl2d.dat
TWIXFILE2=/home/ague/data/T1825/T1825.dat
TWIXFILE3=/home/ague/archive/pha/2018/2018-04-13_RT-Export/U00000013.dat
TWIXFILE4=/home/ague/archive/pha/2019/2019-03-25_Other-MU_Export/meas_MID00311_FID65562_t1_fl2d.dat
TWIXFILE5=/home/ague/archive/vol/2020-02-27_MRT5_MyT1_0038/meas_MID00044_FID160057_UE02_UMG_Radial_ssl_fully_mulpIR.dat
TWIXFILE6=/home/ague/archive/pha/2019/2019-07-24_QuantMapping//meas_MID00146_FID98948_2019_07_24_UMG_Radial_f424a43_RebasedInv_FA45_FS_BR192_Meas1000.dat

tests/test-twixread: twixread ${TWIXFILE1}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIXFILE1} ksp.ra					;\
	echo "c304bf3bb41c7571408e776ec1280870 *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-multiraid: twixread ${TWIXFILE4}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIXFILE1} ksp.ra					;\
	echo "c304bf3bb41c7571408e776ec1280870 *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-multiinversion: twixread ${TWIXFILE5}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -x512 -r15 -c16 -n100 -i30 ${TWIXFILE5} ksp.ra					;\
	echo "41507b81f2861e872dae5465f22fbbbe *ksp.ra" | md5sum -c			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-twixread-multiinversion_mibSSFP: twixread ${TWIXFILE6}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -x384 -B -r1 -c20 -n1000 -i192 ${TWIXFILE6} ksp.ra					;\
	echo "fbc5bbb397d34012cfba4590d9d57182 *ksp.ra" | md5sum -c			;\
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



TESTS_AGUE += tests/test-twixread tests/test-twixread-multiraid tests/test-twixread-mpi tests/test-umgread tests/test-twixread-multiinversion
TESTS_AGUE += tests/test-twixread-multiinversion_mibSSFP
