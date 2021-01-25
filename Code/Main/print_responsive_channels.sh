# Print responsive and amodoal channels for all cases

data_type='micro'
level='sentence_offset'
filter='gaussian-kernel'
for patient in '479_11' '479_25' '482' '487' '493' '502' '504' '505' '510' '513' '515'; do
    python print_responsive_channels.py --patient $patient --data-type $data_type --level $level --filter $filter
done
    
#for patient in '479_11' '479_25' '482' '487' '493' '502' '504' '505' '510' '513' '515'; do
#    for data_type in 'micro' 'macro';do
#        for filter in 'raw' 'gaussian-kernel' 'high-gamma';do
#            for level in 'phone' 'word' 'sentence_onset' 'sentence_offset';do
#                python print_responsive_channels.py --patient $patient --data-type $data_type --level $level --filter $filter
#            done
#        done
#    done
#done
