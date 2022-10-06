# Local(0) or Alambic (1)?
CLUSTER=1

#
METHODS='remove'
DECIMATE=50
SMOOTH=50

PATIENTS="479_11 479_25 482 499 502 505 510 513 515 530 538 539 540 541 543 544 545 549 551 552 553 554_4 554_13"
#PATIENTS="479_11 502 510 513"
DTYPES_FILTERS="micro_raw micro_high-gamma macro_raw macro_high-gamma"
EACH=" --each-feature-value"


queue="Unicog_long"
walltime="02:00:00"


BLOCKS='visual auditory'
for BLOCK in $BLOCKS; do
    if [ $BLOCK == "auditory" ]
    then
    FEATURES="boundaries phonemes lexicon glove syntax"
    QTRAIN="'block in [2,4,6] and word_length>1'"
    QTEST="'block in [2,4,6] and word_length>1'"
    else
    FEATURES="boundaries orthography lexicon glove syntax"
    QTRAIN="'block in [1,3,5] and word_length>1'"
    QTEST="'block in [1,3,5] and word_length>1'"
    fi

	for METHOD in $METHODS; do
	    for PATIENT in $PATIENTS; do
		for DTYPE_FILTER in $DTYPES_FILTERS; do
			DTYPE=${DTYPE_FILTER%%_*}
			FILTER=${DTYPE_FILTER##*_}
        for FEATURE in $FEATURES; do
		FLIST="position "$FEATURE
                #FLIST=$FEATURE
                    path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/encoding/"

                    filename_py="plot_encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --feature-list "$FLIST" --query-train "$QTRAIN" --query-test "$QTEST" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE$EACH

                    output_log='logs/out_plot_trf_'$PATIENT'_'$DTYPE'_'$FILTER
                    error_log='logs/err_plot_trf_'$PATIENT'_'$DTYPE'_'$FILTER
                    job_name='plotTRF_'$PATIENT'_'$DTYPE'_'$FILTER

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
done
