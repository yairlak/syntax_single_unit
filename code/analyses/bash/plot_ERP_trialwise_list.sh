# 
#MAT=" --from-mat"
MAT=""
# Which patients to run (e.g., 479_11 479_25 482 487 493 502 504 505 510 513 515)?
PATIENTS="479_11 479_25 482 489 493 499 502 504 505 510 513 515 530 538 539 540"
PATIENTS="541"

# Which signal types (micro macro spike)
DTYPES="micro macro spike microphone"
DTYPES="spike"

# Which filter (raw high-gamma)?
FILTERS="raw high-gamma"
FILTERS="raw"

LEVELS="sentence_onset sentence_offset"
COMPARISONS="all_trials all_trials_chrono all_end_trials"

#LEVELS="word"
#COMPARISONS="all_words"

# Local(0) or Alambic (1)?
CLUSTER=1

queue="Unicog_long"
walltime="02:00:00"



for PATIENT in $PATIENTS; do
    for DTYPE in $DTYPES; do
        for LEVEL in $LEVELS; do
            for FILTER in $FILTERS; do
                for COMPARISON in $COMPARISONS; do
                            if [ $FILTER == 'high-gamma' ]
			    then
			        #BASELINE=' --baseline "(None, 0)" --baseline-mode zlogratio'
			        BASELINE=''
			    else
			        BASELINE=''
			    fi
			    path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"
			    filename_py='plot_ERP_trialwise.py --patient '$PATIENT' --data-type '$DTYPE' --level '$LEVEL' --filter '$FILTER' --comparison-name '$COMPARISON$BASELINE$RESP_FLAG$BLOCK
                #' --sort-key '$SORT
			    output_log='logs/out_plot_ERP_trialwise_'$PATIENT'_'$DTYPE'_'$LEVEL'_'$FILTER'_'$COMPARISON'_'$BLOCK
			    error_log='logs/err_plot_ERP_trialwise_'$PATIENT'_'$DTYPE'_'$LEVEL'_'$FILTER'_'$COMPARISON'_'$BLOCK
			    job_name=$PATIENT'_'$DTYPE'_'$LEVEL'_'$FILTER'_'$COMPARISON'_'$BLOCK

			    CMD="python $path2script$filename_py"

			    if [ $CLUSTER -eq 1 ]
			    then
				echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
			    else
				echo $CMD' &' # 1>'$output_log' 2>'$error_log' &'
			    fi
                done
            done
        done
    done
done


