
for patient in '504' '510' '479_11' '479_25' '482' '487' '493' '502' '505' '513' '515'; do # ALL patients
#for patient in '479_11' '479_25' '482' '487' '493' '502' '505' '513' '515'; do # Neuralynx only
#for patient in '504' '510'; do # BlackRock only
    for data_type in 'micro' 'macro'; do
        matlab -nodisplay -nodesktop -nosplash -r "extract_high_gamma('$patient', '$data_type')" #& 1>logs/extract_highgamma_$patient-$data_type.out 2>logs/extract_highgamma_$patient-$data_type.err 
    done
done
