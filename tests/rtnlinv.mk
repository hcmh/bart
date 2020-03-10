


tests/test-rtnlinv-noncart-pi: traj scale phantom rtnlinv fmac nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                                         ;\
	$(TOOLDIR)/traj -r -x256 -y45 -t5 traj.ra                                             ;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra                                                 ;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra                                             ;\
	$(TOOLDIR)/rtnlinv -N -S -o1 -i7 -t traj.ra ksp.ra r.ra c.ra			      ;\
	$(TOOLDIR)/fmac r.ra c.ra x.ra                                                        ;\
	$(TOOLDIR)/nufft traj.ra x.ra ksp2.ra                                                 ;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra ksp2.ra                                               ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-rtnlinv-noncart-pi
