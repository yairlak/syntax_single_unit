thresh=0.005
for block_type in 'visual' 'auditory';do
    for data_type in 'micro' 'macro' 'spike';do
       for filter in 'gaussian-kernel';do
            for level in 'sentence_onset';do
                echo python viz_latencies.py --patients 479_11 479_25 482 487 493 502 504 505 510 513 515 --data-type $data_type --filter $filter --level $level --thresh $thresh --block-type $block_type
                #echo python viz_latencies.py --patients 479_11 --data-type $data_type --filter $filter --level $level --thresh $thresh --block-type $block_type
            done
        done
    done
done
