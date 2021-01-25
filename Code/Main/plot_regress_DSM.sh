# 
LANG=en_US

model_type='logistic'
min_trials=15
time_window=0.4
num_bins=6


times=''
for T in $(seq 0 0.025 0.3); do
 times+=" $T"
done

echo python plot_regress_DSM.py --level word --patient 493 --data-type micro --filter gaussian-kernel --probe-name RFSG --patient 502 --data-type micro --filter gaussian-kernel --probe-name RFSG LFSG --patient 505 --data-type micro --filter gaussian-kernel --probe-name LFGA LFGP --patient 515 --data-type micro --filter gaussian-kernel --probe-name LFSG RFSG --patient 493 --data-type spike --filter gaussian-kernel --probe-name RFSG --patient 502 --data-type spike --filter gaussian-kernel --probe-name RFSG LFSG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LFGA LFGP --patient 515 --data-type spike --filter gaussian-kernel --probe-name LFSG --patient 502 --data-type macro --filter gaussian-kernel --probe-name RFSG LFSG --patient 505 --data-type macro --filter gaussian-kernel --probe-name LFGP --patient 515 --data-type macro --filter gaussian-kernel --probe-name LFSG RFSG --comparison-name word_string --block-type auditory --min-trials $min_trials --time-window $time_window --num-bins $num_bins --times $times --model-type $model_type


#echo python plot_regress_DSM.py --level word --patient 493 --data-type micro --filter gaussian-kernel --probe-name RFSG --patient 502 --data-type micro --filter gaussian-kernel --probe-name RFSG LFSG --patient 505 --data-type micro --filter gaussian-kernel --probe-name LFGA LFGP --patient 515 --data-type micro --filter gaussian-kernel --probe-name LFSG RFSG --comparison-name word_string --block-type auditory --min-trials 15 --time-window 0.4 --num-bins 8 --times $times --model-type logistic

#python plot_regress_DSM.py --level phone --patient 479_11 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type micro --filter high-gamma --probe-name LSTG --patient 487 --data-type micro --filter high-gamma --probe-name LSTG --patient 505 --data-type micro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type micro --filter high-gamma --probe-name LSTG --comparison-name phone --times $times --responsive-channels-only &

#python plot_regress_DSM.py --level phone --patient 479_11 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type macro --filter high-gamma --probe-name LSTG --patient 487 --data-type macro --filter high-gamma --probe-name LSTG --patient 505 --data-type macro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type macro --filter high-gamma --probe-name LSTG --comparison-name phone --times $times --responsive-channels-only &

#python plot_regress_DSM.py --level phone --patient 479_11 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LHSG LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel --probe-name LSTG --comparison-name phone --times $times --responsive-channels-only &

#echo python plot_regress_DSM.py --level phone --patient 479_11 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type micro --filter high-gamma --probe-name LSTG --patient 487 --data-type micro --filter high-gamma --probe-name LSTG --patient 505 --data-type micro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type micro --filter high-gamma --probe-name LSTG --comparison-name phone --times $times --responsive-channels-only &

#echo python plot_regress_DSM.py --level phone --patient 479_11 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type macro --filter high-gamma --probe-name LSTG --patient 487 --data-type macro --filter high-gamma --probe-name LSTG --patient 505 --data-type macro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type macro --filter high-gamma --probe-name LSTG --comparison-name phone --times $times --responsive-channels-only &

#echo python plot_regress_DSM.py --level phone --patient 479_11 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LHSG LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel --probe-name LSTG --comparison-name phone --times $times --responsive-channels-only &
