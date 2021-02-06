
# for math in lowmem test
SHELL := /bin/bash

$(TESTS_OUT)/pattern.ra: poisson reshape
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e $(TESTS_TMP)/poisson.ra	;\
	$(TOOLDIR)/reshape 7 32 32 1 poisson.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)


$(TESTS_OUT)/pattern_batch.ra: poisson reshape join
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -s1 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e poisson1.ra		;\
	$(TOOLDIR)/poisson -s2 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e poisson2.ra		;\
	$(TOOLDIR)/poisson -s3 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e poisson3.ra		;\
	$(TOOLDIR)/poisson -s4 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e poisson4.ra		;\
	$(TOOLDIR)/reshape 7 32 32 1 poisson1.ra rpoisson1.ra				;\
	$(TOOLDIR)/reshape 7 32 32 1 poisson2.ra rpoisson2.ra				;\
	$(TOOLDIR)/reshape 7 32 32 1 poisson3.ra rpoisson3.ra				;\
	$(TOOLDIR)/reshape 7 32 32 1 poisson4.ra rpoisson4.ra				;\
	$(TOOLDIR)/join 4 rpoisson1.ra rpoisson2.ra rpoisson3.ra rpoisson4.ra $@	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)	

$(TESTS_OUT)/train_kspace.ra: phantom join scale fmac $(TESTS_OUT)/pattern.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r1 -k kphan1.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r2 -k kphan2.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r3 -k kphan3.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r4 -k kphan4.ra				;\
	$(TOOLDIR)/join 4 kphan1.ra kphan2.ra kphan3.ra kphan4.ra kphan.ra		;\
	$(TOOLDIR)/scale 32 kphan.ra kphan.ra						;\
	$(TOOLDIR)/fmac kphan.ra $(TESTS_OUT)/pattern.ra $@				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)						;\

$(TESTS_OUT)/train_kspace_batch_pattern.ra: phantom join scale fmac $(TESTS_OUT)/pattern_batch.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r1 -k kphan1.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r2 -k kphan2.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r3 -k kphan3.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r4 -k kphan4.ra					;\
	$(TOOLDIR)/join 4 kphan1.ra kphan2.ra kphan3.ra kphan4.ra kphan.ra			;\
	$(TOOLDIR)/scale 32 kphan.ra kphan.ra							;\
	$(TOOLDIR)/fmac kphan.ra $(TESTS_OUT)/pattern_batch.ra $@				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/train_ref.ra: phantom join scale rss fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r1 phan1.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r2 phan2.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r3 phan3.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r4 phan4.ra					;\
	$(TOOLDIR)/join 4 phan1.ra phan2.ra phan3.ra phan4.ra phan.ra			;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/fmac phan.ra scale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)						;\

$(TESTS_OUT)/train_sens.ra: phantom rss invert fmac repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/repmat 4 4 sens.ra sens.ra						;\
	$(TOOLDIR)/invert scale.ra iscale.ra						;\
	$(TOOLDIR)/fmac sens.ra iscale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/test_kspace.ra: phantom join scale fmac $(TESTS_OUT)/pattern.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r5 -k kphan5.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r6 -k kphan6.ra				;\
	$(TOOLDIR)/join 4 kphan5.ra kphan6.ra kphan.ra					;\
	$(TOOLDIR)/scale 32 kphan.ra kphan.ra						;\
	$(TOOLDIR)/fmac kphan.ra $(TESTS_OUT)/pattern.ra $@				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/test_ref.ra: phantom join scale rss fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r5 phan5.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r6 phan6.ra					;\
	$(TOOLDIR)/join 4 phan5.ra phan6.ra phan.ra					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/fmac phan.ra scale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/test_sens.ra: phantom rss invert fmac repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/repmat 4 2 sens.ra sens.ra						;\
	$(TOOLDIR)/invert scale.ra iscale.ra						;\
	$(TOOLDIR)/fmac sens.ra iscale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)


tests/test-nnvn-train: nrmse $(TESTS_OUT)/pattern.ra nnvn \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/nnvn --test_defaults -i -n $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra $(TESTS_OUT)/pattern.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/nnvn --test_defaults -i -n -t -e20 -b2 $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra $(TESTS_OUT)/pattern.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/nnvn --test_defaults -a -n $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra $(TESTS_OUT)/pattern.ra weights0 out0.ra					;\
	$(TOOLDIR)/nnvn --test_defaults -a -n $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra $(TESTS_OUT)/pattern.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.3 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nnmodl-train: nrmse $(TESTS_OUT)/pattern.ra nnmodl \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/nnmodl --test_defaults -i -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/nnmodl --test_defaults -i -n -t -e2 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/nnmodl --test_defaults -lweights01 -n -t -e10 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/nnmodl --test_defaults -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/nnmodl --test_defaults -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nnvn-train-batch-pattern: nrmse $(TESTS_OUT)/pattern.ra $(TESTS_OUT)/pattern_batch.ra nnvn \
	$(TESTS_OUT)/train_kspace_batch_pattern.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/nnvn --test_defaults -i -n $(TESTS_OUT)/train_kspace_batch_pattern.ra $(TESTS_OUT)/train_sens.ra $(TESTS_OUT)/pattern.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/nnvn --test_defaults -i -n -t -e20 -b2 $(TESTS_OUT)/train_kspace_batch_pattern.ra $(TESTS_OUT)/train_sens.ra $(TESTS_OUT)/pattern_batch.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/nnvn --test_defaults -a -n $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra $(TESTS_OUT)/pattern.ra weights0 out0.ra					;\
	$(TOOLDIR)/nnvn --test_defaults -a -n $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra $(TESTS_OUT)/pattern.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.3 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nnvn-train-gpu: nrmse $(TESTS_OUT)/pattern.ra nnvn \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/nnvn -g --test_defaults -i -n $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra $(TESTS_OUT)/pattern.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/nnvn -g --test_defaults -i -n -t -e20 -b2 $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra $(TESTS_OUT)/pattern.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/nnvn -g --test_defaults -a -n $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra $(TESTS_OUT)/pattern.ra weights0 out0.ra					;\
	$(TOOLDIR)/nnvn -g --test_defaults -a -n $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra $(TESTS_OUT)/pattern.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.3 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nnmodl-train-gpu: nrmse $(TESTS_OUT)/pattern.ra nnmodl \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/nnmodl -g --test_defaults -i -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/nnmodl -g --test_defaults -i -n -t -e10 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/nnmodl -g --test_defaults -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/nnmodl -g --test_defaults -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-nnmodl-train
TESTS += tests/test-nnvn-train

TESTS_GPU += tests/test-nnmodl-train-gpu
TESTS_GPU += tests/test-nnvn-train-gpu


