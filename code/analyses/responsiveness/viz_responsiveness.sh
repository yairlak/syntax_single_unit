thresh=0.05
for data_type in 'micro' 'macro' 'spike';do
   for filter in 'raw' 'gaussian-kernel' 'high-gamma';do
        for level in 'phone' 'word' 'sentence_onset' 'sentence_offset';do
            python viz_responsiveness.py --patients 479_11 479_25 482 487 493 502 504 505 510 513 515 --data-type $data_type --filter $filter --level $level --thresh $thresh
        done
    done
done
