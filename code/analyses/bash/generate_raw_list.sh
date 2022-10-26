# 
# Local(0) or Alambic (1)?
CLUSTER=0

#MAT=" --from-mat"
MAT=""
# Which patients to run (e.g., 479_11 479_25 482 487 493 502 504 505 510 513 515)?
PATIENTS="479_11 479_25 482 499 502 505 513 515 538 540 541 543 545 549 551 552 553 554_4 554_13 510 530 539 544 556" # NEURALYNX AND BLACKROCK
# PATIENTS="479_11 479_25 482 499 502 505 513 515 538 540 541 543 545 549 551 552 553 554_4 554_13" # NEURALYNX MICRO
# PATIENTS="510 530 539 544 556" # BLACKROCK MICRO
# PATIENTS="545"

# Which signal types (micro macro spike)
# Which filter (raw high-gamma)?
DTYPES_FILTERS="micro_raw micro_high-gamma macro_raw macro_high-gamma spike_raw microphone_raw"
DTYPES_FILTERS="macro_raw macro_high-gamma"
#DTYPES_FILTERS="micro_raw micro_high-gamma spike_raw microphone_raw"
#DTYPES_FILTERS="micro_raw micro_high-gamma macro_raw macro_high-gamma spike_raw"


#queue="Unicog_long"
#queue="Nspin_bigM"
queue="Nspin_long"
walltime="72:00:00"

for PATIENT in $PATIENTS; do
    for DTYPE_FILTER in $DTYPES_FILTERS; do
            DTYPE=${DTYPE_FILTER%%_*}
            FILTER=${DTYPE_FILTER##*_}
	    #echo $PATIENT $DTYPE $FILTER
	    path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"
	    # !! note if generate_raw_.py or generate_raw.py !!
	    filename_py="generate_raw__.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER$MAT
	    output_log='logs/out_generate_mne_raw_'$PATIENT'_'$DTYPE'_'$FILTER
	    error_log='logs/err_generate_mne_raw_'$PATIENT'_'$DTYPE'_'$FILTER
	    job_name=$PATIENT'_'$DTYPE'_'$FILTER

	    CMD="python3 $path2script$filename_py"

	    if [ $CLUSTER -eq 1 ]
	    then
		echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
	    else
		#echo $CMD' &'
		echo $CMD
		#' 1>'$output_log' 2>'$error_log 
	    fi
    done
done
