# Local(0) or Alambic (1)?
CLUSTER=1

#
CV_FOLDS_IN=5
CV_FOLDS_OUT=5
METHODS='remove'
SMOOTH=50
DECIMATE=50

# Which patients to run (e.g., 479_11 479_25 482 487 493 502 504 505 510 513 515)?
PATIENTS="479_11 479_25 482 499 502 505 510 513 515 530 538 539 540 541 543 544 549 551"
DTYPES_FILTERS="micro_raw micro_high-gamma macro_raw macro_high-gamma spike_raw"

queue="Nspin_bigM"
walltime="72:00:00"

BLOCKS='visual auditory'
for BLOCK in $BLOCKS; do
    if [ $BLOCK == "auditory" ]
    then
    FLIST="position phonology lexicon syntax semantics"
    QTRAIN="'block in [2,4,6] and word_length>1'"
    QTEST="'block in [2,4,6] and word_length>1'"
    else
    FLIST="position orthography lexicon syntax semantics"
    QTRAIN="'block in [1,3,5] and word_length>1'"
    QTEST="'block in [1,3,5] and word_length>1'"
    fi
    for METHOD in $METHODS; do
        for PATIENT in $PATIENTS; do
            for DTYPE_FILTER in $DTYPES_FILTERS; do
                    DTYPE=${DTYPE_FILTER%%_*}
                    FILTER=${DTYPE_FILTER##*_}
                    #echo $PATIENT $DTYPE $FILTER
                    path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"

                    filename_py="encoding_evoked.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE

                    output_log='logs/out_trf_'$PATIENT'_'$DTYPE'_'$FILTER
                    error_log='logs/err_trf_'$PATIENT'_'$DTYPE'_'$FILTER
                    job_name=$PATIENT'_'$DTYPE'_'$FILTER

                    CMD="python3 $path2script$filename_py"

                    if [ $CLUSTER -eq 1 ]
                    then
                        echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
                    else
                        echo $CMD
                        #' 1>'$output_log' 2>'$error_log 
                    fi
            done
        done
    done
done
