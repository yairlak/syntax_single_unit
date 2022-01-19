# Local(0) or Alambic (1)?
CLUSTER=0

#
METHODS='zero remove'
DECIMATE=50
SMOOTH=25
PATIENTS="479_11 479_25 482 489 493 499 502 504 505 510 513 515 530 538 539 540 541"
#PATIENTS="479_11 502 510 513"
EACH=""

# Which signal types (micro macro spike)
#DTYPES="micro macro spike"
DTYPES="micro spike"

# Which filter (raw high-gamma)?
#FILTERS="raw high-gamma"
FILTERS="raw high-gamma"


queue="Nspin_long"
walltime="02:00:00"

BLOCKS='visual auditory'
for BLOCK in $BLOCKS; do
    if [ $BLOCK == "auditory" ]
    then
    FLIST="is_first_word word_onset positional phonology semantics lexicon syntax"
    QTRAIN="'block in [2,4,6] and word_length>1'"
    QTEST="'block in [2,4,6] and word_length>1'"
    else
    FLIST="is_first_word word_onset positional orthography semantics lexicon syntax"
    QTRAIN="'block in [1,3,5] and word_length>1'"
    QTEST="'block in [1,3,5] and word_length>1'"
    fi

for METHOD in $METHODS; do
    for PATIENT in $PATIENTS; do
        for DTYPE in $DTYPES; do
            for FILTER in $FILTERS; do
                # echo $PATIENT $DTYPE $FILTER
                path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/encoding/"

                filename_py="plot_encoding_evoked.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --feature-list "$FLIST" --query-train "$QTRAIN" --query-test "$QTEST" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE$EACH

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
