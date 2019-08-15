
tests/test-mdb-bloch: phantom creal saxpy conj threshold ones fmac cabs join copy bloch fft modbloch slice avg show repmat nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(TOOLDIR)/phantom -x 24 -T map.ra													;\
	$(TOOLDIR)/creal map.ra t1.ra														;\
	$(TOOLDIR)/saxpy -- -1 map.ra t1.ra tmp1.ra												;\
	$(TOOLDIR)/conj tmp1.ra t2.ra														;\
	$(TOOLDIR)/threshold -B 0.01 t1.ra mask.ra												;\
	$(TOOLDIR)/ones 16 24 24 1 1 1 1 1 1 1 1 1 1 1 1 1 1 tmp_m0.ra										;\
	$(TOOLDIR)/fmac tmp_m0.ra mask.ra m0.ra													;\
	$(TOOLDIR)/cabs t2.ra t2_conj.ra													;\
	$(TOOLDIR)/join 6 t1.ra t2_conj.ra m0.ra ref.ra												;\
	$(TOOLDIR)/ones 16 24 24 1 1 1 1000 1 1 1 1 1 1 1 1 1 1 psf.ra										;\
	$(TOOLDIR)/copy m0.ra b1.ra														;\
	$(TOOLDIR)/ones 16 10 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 slice_profile.ra									;\
	$(TOOLDIR)/bloch -R -I t1.ra -i t2.ra -M m0.ra -s1 -f45 -p0.0009 -r1000 -a1 -O -t0.0045 -e0.00225 sig.ra sr1.ra sr2.ra sm0.ra		;\
	$(TOOLDIR)/fft -u 7 sig.ra sig_ksp.ra													;\
	$(TOOLDIR)/modbloch -t0.0045 -e0.00225 -R3 -i15 -f1 -a1 -w0.001 -r0 -I b1.ra -p psf.ra -P slice_profile.ra sig_ksp.ra reco.ra sens.ra	;\
	$(TOOLDIR)/slice 6 2 reco.ra m0reco.ra													;\
	$(TOOLDIR)/avg 7 m0reco.ra mean.ra													;\
	$(TOOLDIR)/threshold -B 0.8171278 m0reco.ra tmp_mask.ra											;\
	$(TOOLDIR)/repmat 6 3 tmp_mask.ra mask2.ra												;\
	$(TOOLDIR)/fmac reco.ra mask2.ra reco_masked.ra												;\
	$(TOOLDIR)/nrmse -t 0.01 ref.ra reco_masked.ra												;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-mdb-bloch
