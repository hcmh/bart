


tests/test-roistat-std: zeros noise ones resize roistat std nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 100 100 3 z.ra						;\
	$(TOOLDIR)/noise -s1. -n1. z.ra n.ra						;\
	$(TOOLDIR)/ones 3 50 50 1 oy.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 oy.ra oy2.ra					;\
	$(TOOLDIR)/roistat -b -D oy2.ra n.ra dy.ra					;\
	$(TOOLDIR)/resize -c 0 50 1 50 n.ra ny2.ra					;\
	$(TOOLDIR)/std 3 ny2.ra dy2.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 dy2.ra dy.ra 					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-roistat-mult: zeros noise ones resize stat join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 100 100 3 z.ra						;\
	$(TOOLDIR)/noise -s1. -n1. z.ra n.ra						;\
	$(TOOLDIR)/ones 3 50 50 1 oy.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 oy.ra oy2.ra					;\
	$(TOOLDIR)/ones 3 70 70 1 ox.ra							;\
	$(TOOLDIR)/resize -c 0 100 1 100 ox.ra ox2.ra					;\
	$(TOOLDIR)/roistat -b -D oy2.ra n.ra dy.ra					;\
	$(TOOLDIR)/roistat -b -D ox2.ra n.ra dx.ra					;\
	$(TOOLDIR)/join 4 dy.ra dx.ra d2.ra 						;\
	$(TOOLDIR)/join 4 oy2.ra ox2.ra o2.ra 						;\
	$(TOOLDIR)/roistat -b -D o2.ra n.ra d.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 d2.ra d.ra 					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-roistat-std tests/test-roistat-mult

