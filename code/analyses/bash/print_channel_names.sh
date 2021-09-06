patients="479_11 479_25 482 489 493 499 502 505 513 515 538"
dtypes="micro macro spike"

for patient in $patients; do
    for dtype in $dtypes; do
        python3 /neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/print_channel_names.py --patient $patient --data-type $dtype
    done
done
