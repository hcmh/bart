
TWIXREAD=/home/ague/archive/2019/2019-03-25_Other-MU_Export/single_raid/meas_MID00311_FID65562_t1_fl2d.dat

tests/test-twixread: twixread ${TWIXREAD}
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/twixread -A ${TWIXREAD} ksp.ra					;\
	echo "c304bf3bb41c7571408e776ec1280870 *ksp.ra" > ksp.md5sum			;\
	md5sum -c ksp.md5sum								;\
	rm ksp.md5sum ksp.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS_AGUE += tests/test-twixread

