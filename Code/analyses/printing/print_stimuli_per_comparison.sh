
for i in $(seq 0 1 47)
do
    filename='comparisons/stimuli_comparison_'$i'.txt'
    rm $filename
    touch $filename
    echo 'Printing comparison '$i
    python print_stimuli_per_comparison.py --comparison $i > $filename
done
