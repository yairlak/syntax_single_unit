
patient='479_11 479_25 482 499 502 505 510 513 515 530 538 539 540 541 543 544 545 549'
#patient='549'

for comparison_name in 'all_trials' 'all_trials_chrono' 'all_end_trials';
do
    for data_type in 'micro' 'macro' 'spike' 'microphone';
    do
        for filter in 'raw' 'high-gamma';
        do
            for level in 'sentence';
            do
                CMD='python3 generate_HTMLs_all_patients_probes.py --patient '$patient' --data-type '$data_type' --filter '$filter' --level '$level' --comparison-name '$comparison_name
                echo $CMD
                eval $CMD
            done
        done
    done
done
