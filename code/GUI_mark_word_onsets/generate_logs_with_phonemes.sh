# BASH

for PATIENT in '479_11' '479_25' '482' '487' '493' '502' '504' '505' '510' '513' '515'; do
    python generate_logs_with_phonemes.py --patient $PATIENT
done
