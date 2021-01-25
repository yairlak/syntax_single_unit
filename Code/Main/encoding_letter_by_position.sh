LANG=en_US
# 
#rm -r RunScripts 
#mkdir RunScripts

echo "Which patients to run (e.g., 479_11 479_25 482 487 493 502 504 505 510 513 515)?"
read PATIENTS

echo "Which signal types (micro macro spike)?"
read DTYPES

echo "Which level (sentence_onset sentence_offset word phone)?"
read LEVELS

echo "Which filter (raw gaussian-kernel gaussian-kernel-25 high-gamma)?"
read FILTERS

echo "Which probes (e.g., RFSG LFSG LFGA RFGP)?"
read PROBES

echo "Which block (auditory or visual)?"
read BLOCK

echo "Which models (e.g., ridge lasso)?"
read MODEL_TYPE

echo "Type starting point for average window (e.g., 0 or -0.05)?"
read TMIN

echo "Type end point for average window (e.g., 0.4 or 0.5)?"
read TMAX

echo "Which  fetures.g., letters or word_length)?"
read FEATURES

echo "Type query for epochs"
read QUERY

echo "Local(0) or Alambic (1)?"
read CLUSTER

if [ $CLUSTER -eq 1 ]
then
    qstat -q

    echo "Choose queue (1: Unicog_long, 2: Nspin_long, 3: Unicog_short, 4: Nspin_short, 5: Unicog_run32, 6: Nspin_run32, 7: Unicog_run16, 8: Nspin_run16, 9:Nspin_bigM)"
    read QUEUE

    if [ $QUEUE -eq 1 ]
    then
        queue="Unicog_long"
        walltime="72:00:00"
    elif [ $QUEUE -eq 2 ]
    then
        queue="Nspin_long"
        walltime="72:00:00"
    elif [ $QUEUE -eq 3 ]
    then
        queue="Unicog_short"
        walltime="02:00:00"
    elif [ $QUEUE -eq 4 ]
    then
        queue="Nspin_short"
        walltime="02:00:00"
    elif [ $QUEUE -eq 5 ]
    then
        queue="Unicog_run32"
        walltime="02:00:00"
    elif [ $QUEUE -eq 6 ]
    then
        queue="Nspin_run32"
        walltime="02:00:00"
    elif [ $QUEUE -eq 7 ]
    then
        queue="Unicog_run16"
        walltime="02:00:00"
    elif [ $QUEUE -eq 8 ]
    then
        queue="Nspin_run16"
        walltime="02:00:00"
    elif [ $QUEUE -eq 9 ]
    then
        queue="Nspin_bigM"
        walltime="72:00:00"
    fi
fi



for patient in $PATIENTS; do
    for data_type in $DTYPES; do
         for level in $LEVELS; do
             for filter in $FILTERS; do
                 for probe_name in $PROBES; do


                    CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/encoding_letter_by_position.py --level '$level' --patient '$patient' --data-type '$data_type' --filter '$filter' --probe-name '$probe_name' --tmin '$TMIN' --tmax '$TMAX' --model-type '$MODEL_TYPE' --query '$QUERY' --block-type '$BLOCK' --feature-list'

                    #for feature in $FEATURES; do
                    #    CMD+=' '$feature
                    #done

                    output_log='logs/rsa_'$model_type-$min_trials-$time_window-$num_bins-$T'.out'
                    error_log='logs/rsa_'$model_type-$min_trials-$time_window-$num_bins-$T'.err'
                    job_name='rsa_'$model_type-$min_trials-$time_window-$num_bins-$T


                    if [ $CLUSTER -eq 1 ]
                    then
                        echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
                    else
                        echo $CMD' 1>'$output_log' 2>'$error_log' &'
                    fi
                done
            done
        done
    done
done
