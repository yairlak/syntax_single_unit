LC_NUMERIC="en_US.UTF-8"

model_type='logistic'
min_trials=15
time_window=0.4
num_bins=6
for T in $(seq 0 0.025 0.3); do
	python regress_DSM.py --level word --patient 493 --data-type micro --filter gaussian-kernel --probe-name RFSG --patient 502 --data-type micro --filter gaussian-kernel --probe-name RFSG LFSG --patient 505 --data-type micro --filter gaussian-kernel --probe-name LFGA LFGP --patient 515 --data-type micro --filter gaussian-kernel --probe-name LFSG RFSG --patient 493 --data-type spike --filter gaussian-kernel --probe-name RFSG --patient 502 --data-type spike --filter gaussian-kernel --probe-name RFSG LFSG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LFGA LFGP --patient 515 --data-type spike --filter gaussian-kernel --probe-name LFSG --patient 502 --data-type macro --filter gaussian-kernel --probe-name RFSG LFSG --patient 505 --data-type macro --filter gaussian-kernel --probe-name LFGP --patient 515 --data-type macro --filter gaussian-kernel --probe-name LFSG RFSG --comparison-name word_string --block-type auditory --min-trials $min_trials --time-window $time_window --num-bins $num_bins --times $T --model-type $model_type &	

done

    

	#echo python rsa_regress.py --level phone --patient 479_11 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type micro --filter high-gamma --probe-name LSTG --patient 487 --data-type micro --filter high-gamma --probe-name LSTG --patient 505 --data-type micro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type micro --filter high-gamma --probe-name LSTG --comparison-name phone --responsive-channels-only --num-bins 4 --times $T --pick-classes B D G P T K M N NG V Z SH S F TH DH R L Y W --path2features functions/phone_features.csv --pick-features DORSAL CORONAL LABIAL PLOSIVE FRICATIVE NASAL VOICED OBSTRUENT &

	#echo python rsa_regress.py --level phone --patient 479_11 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type macro --filter high-gamma --probe-name LSTG --patient 487 --data-type macro --filter high-gamma --probe-name LSTG --patient 505 --data-type macro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type macro --filter high-gamma --probe-name LSTG --comparison-name phone --responsive-channels-only --num-bins 4 --times $T --pick-classes B D G P T K M N NG V Z SH S F TH DH R L Y W --path2features functions/phone_features.csv --pick-features DORSAL CORONAL LABIAL PLOSIVE FRICATIVE NASAL VOICED OBSTRUENT &

	#echo python rsa_regress.py --level phone --patient 479_11 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LHSG LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel --probe-name LSTG --comparison-name phone --responsive-channels-only --num-bins 4 --times $T --pick-classes B D G P T K M N NG V Z SH S F TH DH R L Y W --path2features functions/phone_features.csv --pick-features DORSAL CORONAL LABIAL PLOSIVE FRICATIVE NASAL VOICED OBSTRUENT &



#python rsa_regress.py --level phone --patient 479_11 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type micro --filter high-gamma --probe-name LSTG --patient 487 --data-type micro --filter high-gamma --probe-name LSTG --patient 505 --data-type micro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type micro --filter high-gamma --probe-name LSTG --comparison-name phone --responsive-channels-only --num-bins 4 --times $T --class-names B D G P T K M N NG V Z SH S F TH DH R L Y W AY AW AE EH EY IH IY UW UX AX OW AO AA --path2features functions/phone_features.csv &
