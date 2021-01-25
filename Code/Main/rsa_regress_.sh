LANG=en_US
# 
#rm -r RunScripts 
#mkdir RunScripts

#echo "Which patients to run (e.g., 479_11 479_25 482 487 493 502 504 505 510 513 515)?"
#read PATIENTS

#echo "Which signal types (micro macro spike)?"
#read DTYPES

#echo "Which level (sentence_onset sentence_offset word phone)?"
#read LEVELS
#
#echo "Which filter (raw gaussian-kernel high-gamma)?"
#read FILTERS

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

#for PATIENT in $PATIENTS; do
#    for DTYPE in $DTYPES; do
#        for LEVEL in $LEVELS; do
#		for FILTER in $FILTERS; do
for T in $(seq 0 0.01 0.5); do
    output_log='logs/out_rsa_regress_'$T
    error_log='logs/err_rsa_regress_'$T
    job_name='rsa_'$T


    if [ $CLUSTER -eq 1 ]
    then
        echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
    else
        echo $CMD' 1>'$output_log' 2>'$error_log' &'
    fi
    CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/rsa_regress.py --level phone --patient 479_11 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type micro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type micro --filter high-gamma --probe-name LSTG --patient 487 --data-type micro --filter high-gamma --probe-name LSTG --patient 505 --data-type micro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type micro --filter high-gamma --probe-name LSTG --comparison-name phone --responsive-channels-only --num-bins 4 --times '$T' --pick-classes B D G P T K M N NG V Z SH S F TH DH R L Y W --path2features functions/phone_features.csv --pick-features DORSAL CORONAL LABIAL PLOSIVE FRICATIVE NASAL VOICED OBSTRUENT'

    output_log='logs/out_rsa_regress_'$T
    error_log='logs/err_rsa_regress_'$T
    job_name='rsa_'$T


    if [ $CLUSTER -eq 1 ]
    then
        echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
    else
        echo $CMD' 1>'$output_log' 2>'$error_log' &'
    fi
    CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/rsa_regress.py --level phone --patient 479_11 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 479_25 --data-type macro --filter high-gamma --probe-name LSTG RPSTG --patient 482 --data-type macro --filter high-gamma --probe-name LSTG --patient 487 --data-type macro --filter high-gamma --probe-name LSTG --patient 505 --data-type macro --filter high-gamma --probe-name LHSG LSTG RSTG --patient 515 --data-type macro --filter high-gamma --probe-name LSTG --comparison-name phone --responsive-channels-only --num-bins 4 --times '$T' --pick-classes B D G P T K M N NG V Z SH S F TH DH R L Y W --path2features functions/phone_features.csv --pick-features DORSAL CORONAL LABIAL PLOSIVE FRICATIVE NASAL VOICED OBSTRUENT'

    output_log='logs/out_rsa_regress_'$T
    error_log='logs/err_rsa_regress_'$T
    job_name='rsa_'$T


    if [ $CLUSTER -eq 1 ]
    then
        echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
    else
        echo $CMD' 1>'$output_log' 2>'$error_log' &'
    fi
    CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/rsa_regress.py --level phone --patient 479_11 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LHSG LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel --probe-name LSTG --comparison-name phone --responsive-channels-only --num-bins 4 --times '$T' --pick-classes B D G P T K M N NG V Z SH S F TH DH R L Y W --path2features functions/phone_features.csv --pick-features DORSAL CORONAL LABIAL PLOSIVE FRICATIVE NASAL VOICED OBSTRUENT'

#		done
#        done     
#    done
done



