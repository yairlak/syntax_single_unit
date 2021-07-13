# 
#rm -r RunScripts 
#mkdir RunScripts

echo "Which patients to run (e.g., 479_11 479_25 482 487 493 502 504 505 510 513 515)?"
read PATIENTS

echo "Which signal types (micro macro spike)?"
read DTYPES

echo "Which level (sentence_onset sentence_offset word phone)?"
read LEVELS

echo "Which filter (raw gaussian-kernel high-gamma)?"
read FILTERS

echo "Which comparison (all_words manner_of_articulation)?"
read COMPARISONS

echo "Which block (auditory visual)?"
read BLOCKS

#echo "Which sorting (e.g., [phone_string])."
#read SORT
#Leave blank for default (sentence length)."

echo "Responsive channels only (yes/no)"
read RESP

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

if [ $RESP == 'yes' ]
then
RESP_FLAG=" --responsive-channels-only"
elif [ $RESP == 'no' ]
then
RESP_FLAG=""
else
echo "WARNING! must be yes or no"
fi

if [ $BLOCKS == 'auditory' ]
then
BLOCK=" --block-type auditory"
elif [ $BLOCKS == 'visual' ]
then
BLOCK=" --block-type visual"
else
BLOCK=""
echo "WARNING! wrong choice of block type (either empty, visual or auditory)"
fi


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
				echo $CMD' 1>'$output_log' 2>'$error_log' &'
			    fi
                done
            done
        done     
    done
done
