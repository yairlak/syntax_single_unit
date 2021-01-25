
patient='479_11 479_25 482 487 493 502 504 505 510 513 515'

for comparison_name in 'all_end_trials' 'all_trials';
do
    for data_type in 'micro' 'spike'; # 'macro';
    do
        for filter in 'raw' 'gaussian-kernel' 'gaussian-kernel-25' 'high-gamma';
        do
            for level in 'sentence_onset' 'sentence_offset';
            do
                CMD='python generate_HTMLs_all_patients_probes.py --patient '$patient' --data-type '$data_type' --filter '$filter' --level '$level' --comparison-name '$comparison_name
                echo $CMD
                eval $CMD
            done
        done
    done
done
