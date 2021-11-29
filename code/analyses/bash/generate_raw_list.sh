# 
#MAT=" --from-mat"
MAT=""
# Which patients to run (e.g., 479_11 479_25 482 487 493 502 504 505 510 513 515)?
#PATIENTS="479_11 479_25 482 489 493 499 502 504 505 510 513 515 530 538 539"
#PATIENTS="479_11 479_25 482 489 493 499 505 513 515 538"
PATIENTS="499"
# Which signal types (micro macro spike)
DTYPES="micro macro spike microphone"
DTYPES="spike micro macro"

# Which filter (raw high-gamma)?
FILTERS="raw high-gamma"

# Local(0) or Alambic (1)?
CLUSTER=0

queue="Nspin_long"
walltime="02:00:00"

for PATIENT in $PATIENTS; do
    for DTYPE in $DTYPES; do
        for FILTER in $FILTERS; do
            echo $PATIENT $DTYPE $FILTER
            path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"
            filename_py="generate_raw.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER$MAT
            output_log='logs/out_generate_mne_raw_'$PATIENT'_'$DTYPE'_'$FILTER
            error_log='logs/err_generate_mne_raw_'$PATIENT'_'$DTYPE'_'$FILTER
            job_name=$PATIENT'_'$DTYPE'_'$FILTER

            CMD="python3 $path2script$filename_py"

            if [ $CLUSTER -eq 1 ]
            then
                echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
            else
                echo $CMD' &'
                #' 1>'$output_log' 2>'$error_log 
            fi
        done 
    done
done
