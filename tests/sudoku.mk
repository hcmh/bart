
#hard but does not work
#bart vec 1 0 0  1 0 1  0 0 1 p4

tests/test-sudoku: vec join fmac sudoku nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/vec 5 3 4  6 7 8  9 1 2 r0.ra					;\
	$(TOOLDIR)/vec 6 7 2  1 9 5  3 4 8 r1.ra					;\
	$(TOOLDIR)/vec 1 9 8  3 4 2  5 6 7 r2.ra					;\
	$(TOOLDIR)/vec 8 5 9  7 6 1  4 2 3 r3.ra					;\
	$(TOOLDIR)/vec 4 2 6  8 5 3  7 9 1 r4.ra					;\
	$(TOOLDIR)/vec 7 1 3  9 2 4  8 5 6 r5.ra					;\
	$(TOOLDIR)/vec 9 6 1  5 3 7  2 8 4 r6.ra					;\
	$(TOOLDIR)/vec 2 8 7  4 1 9  6 3 5 r7.ra					;\
	$(TOOLDIR)/vec 3 4 5  2 8 6  1 7 9 r8.ra					;\
	$(TOOLDIR)/join 1 r0.ra r1.ra r2.ra r3.ra r4.ra r5.ra r6.ra r7.ra r8.ra board.ra;\
	$(TOOLDIR)/vec 1 1 0  0 1 0  0 0 0 p0.ra					;\
	$(TOOLDIR)/vec 1 0 0  1 1 1  0 0 0 p1.ra					;\
	$(TOOLDIR)/vec 0 1 1  0 0 0  0 1 0 p2.ra					;\
	$(TOOLDIR)/vec 1 0 0  0 1 0  0 0 1 p3.ra					;\
	$(TOOLDIR)/vec 1 1 0  1 0 1  0 0 1 p4.ra					;\
	$(TOOLDIR)/vec 1 0 0  0 1 0  0 0 1 p5.ra					;\
	$(TOOLDIR)/vec 0 1 0  0 0 0  1 1 0 p6.ra					;\
	$(TOOLDIR)/vec 0 0 0  1 1 1  0 0 1 p7.ra					;\
	$(TOOLDIR)/vec 0 0 0  0 1 0  0 1 1 p8.ra					;\
	$(TOOLDIR)/join 1 p0.ra p1.ra p2.ra p3.ra p4.ra p5.ra p6.ra p7.ra p8.ra pat.ra	;\
	$(TOOLDIR)/fmac board.ra pat.ra und.ra						;\
	$(TOOLDIR)/sudoku -l0. und.ra solution.ra					;\
	$(TOOLDIR)/nrmse -t 0. board.ra solution.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-sudoku


