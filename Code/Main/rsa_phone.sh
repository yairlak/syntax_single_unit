LANG=en_US
# 
#pick_classes='B D G P T K M N NG V Z SH S F TH DH R L Y W AY AW AE EH EY IH IY UW UX AX OW AO AA'
#pick_classes='B D G P T K M N NG V Z SH S F DH R L W AH IY IH EH'
pick_classes='B D G P T K M N NG'
responsive='' # either '' or ' --responsive-channels-only' 

echo "Which models (e.g., logistic lstm cnn)?"
read MODELs

echo "Which starting points (e.g., 0 0.05 0.1 0.15 0.2)?"
read Ts

echo "Which time windows (e.g., 0.1 0.2 0.3 0.4 0.5)?"
read TIME_WINDOWs

echo "Which numbers of bins (e.g., 1 2 4 8 16 32)?"
read NUM_BINs

echo "Which minimum numbers of trials per class (e.g., 3 9 15 21)?"
read MIN_TRIALs

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

for model_type in $MODELs; do
    for min_trials in $MIN_TRIALs; do
        for time_window in $TIME_WINDOWs; do
            for num_bins in $NUM_BINs; do
                for T in $Ts; do
					output_log='logs/out_rsa_regress_'$T
					error_log='logs/err_rsa_regress_'$T
					job_name='rsa_'$T
					
					CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/rsa.py --level phone --patient 479_11 --data-type micro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type micro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type micro --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type micro --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type micro --filter gaussian-kernel --probe-name LSTG RSTG --patient 515 --data-type micro --filter gaussian-kernel --probe-name LSTG --comparison-name phone --min-trials '$min_trials' --time-window '$time_window' --num-bins '$num_bins' --times '$T' --pick-classes '$pick_classes' --model-type '$model_type$responsive


					if [ $CLUSTER -eq 1 ]
					then
						echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
					else
						echo $CMD' 1>'$output_log' 2>'$error_log' &'
					fi

					CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/rsa.py --level phone --patient 479_11 --data-type macro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type macro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type macro --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type macro --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type macro --filter gaussian-kernel --probe-name LSTG RSTG --patient 515 --data-type macro --filter gaussian-kernel --probe-name LSTG --comparison-name phone --min-trials '$min_trials' --time-window '$time_window' --num-bins '$num_bins' --times '$T' --pick-classes '$pick_classes' --model-type '$model_type


					if [ $CLUSTER -eq 1 ]
					then
						echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
					else
						echo $CMD' 1>'$output_log' 2>'$error_log' &'
					fi
					
					CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/rsa.py --level phone --patient 479_11 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel --probe-name LSTG --comparison-name phone --min-trials '$min_trials' --time-window '$time_window' --num-bins '$num_bins' --times '$T' --pick-classes '$pick_classes' --model-type '$model_type$responsive

					if [ $CLUSTER -eq 1 ]
					then
						echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
					else
						echo $CMD' 1>'$output_log' 2>'$error_log' &'
					fi

					CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/rsa.py --level phone --patient 479_11 --data-type micro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type micro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type micro --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type micro --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type micro --filter gaussian-kernel --probe-name LSTG RSTG --patient 515 --data-type micro --filter gaussian-kernel --probe-name LSTG --patient 479_11 --data-type macro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type macro --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type macro --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type macro --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type macro --filter gaussian-kernel --probe-name LSTG RSTG --patient 515 --data-type macro --filter gaussian-kernel --probe-name LSTG --patient 479_11 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel --probe-name LSTG --comparison-name phone --responsive-channels-only --min-trials '$min_trials' --time-window '$time_window' --num-bins '$num_bins' --times '$T' --pick-classes '$pick_classes' --model-type '$model_type


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


