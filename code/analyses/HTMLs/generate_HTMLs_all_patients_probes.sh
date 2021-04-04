
patient='479_11 479_25 482 487 493 502 504 505 510 513 515'
#patient='487'

#for comparison_name in 'all_end_trials' 'all_trials';
for comparison_name in 'all_trials';
do
    for data_type in 'micro' 'macro' 'spike';
    #for data_type in 'macro';
    do
        for filter in 'raw' 'gaussian-kernel-10' 'gaussian-kernel-25' 'high-gamma';
        #for filter in 'gaussian-kernel-10' 'gaussian-kernel-25';
        do
            #for level in 'sentence_onset' 'sentence_offset';
            for level in 'sentence_onset';
            do
                CMD='python generate_HTMLs_all_patients_probes.py --patient '$patient' --data-type '$data_type' --filter '$filter' --level '$level' --comparison-name '$comparison_name
                echo $CMD
                eval $CMD
            done
        done
    done
done
