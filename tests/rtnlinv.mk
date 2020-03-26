

# test rtnlinv when using gridding Y and P
tests/test-rtnlinv-noncart-pi: traj scale phantom nufft rtnlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                ;\
	$(TOOLDIR)/traj -r -x128 -y21 -t5 traj.ra                    ;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra                        ;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra                    ;\
	$(TOOLDIR)/nufft -a traj.ra ksp.ra I.ra                      ;\
	$(TOOLDIR)/rtnlinv -N -S -o1 -i7 -t traj.ra ksp.ra r.ra c.ra ;\
	$(TOOLDIR)/fmac r.ra c.ra x.ra                               ;\
	$(TOOLDIR)/nufft traj.ra x.ra k2.ra                          ;\
	$(TOOLDIR)/nufft -a traj.ra k2.ra x2.ra                      ;\
	$(TOOLDIR)/nrmse -t 0.05 I.ra x2.ra                          ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_SLOW += tests/test-rtnlinv-noncart-pi
