# Local(0) or Alambic (1)?
CLUSTER=1

# PARAMS
CV_FOLDS_IN=5
CV_FOLDS_OUT=5
METHOD='zero'
SMOOTH=25
DECIMATE=50

# CLUSTER
queue="Nspin_long"
walltime="72:00:00"

# PATH
path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"

##############
# CASE STUDY #
##############
PATIENT='479_11'
DTYPE='spike'
FILTER='raw'
QTRAIN="'block in [2,4,6] and word_length>1'"
QTEST="'block in [2,4,6] and word_length>1'"
FLIST="is_first_word word_onset positional phonology"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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


##############
# CASE STUDY #
##############
PATIENT='505'
DTYPE='spike'
FILTER='raw'
FLIST="is_first_word word_onset positional orthography"
QTRAIN="'block in [1,3,5] and word_length>1'"
QTEST="'block in [1,3,5] and word_length>1'"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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


##############
# CASE STUDY #
##############
PATIENT='510'
DTYPE='micro'
FILTER='raw'
QTRAIN="'block in [2,4,6] and word_length>1'"
QTEST="'block in [2,4,6] and word_length>1'"
FLIST="is_first_word word_onset positional syntax"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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

##############
# CASE STUDY #
##############
PATIENT='510'
DTYPE='micro'
FILTER='raw'
QTRAIN="'block in [1,3,5] and word_length>1'"
QTEST="'block in [1,3,5] and word_length>1'"
FLIST="is_first_word word_onset positional syntax"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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

##############
# CASE STUDY #
##############
PATIENT='510'
DTYPE='micro'
FILTER='raw'
QTRAIN="'block in [2,4,6] and word_length>1'"
QTEST="'block in [2,4,6] and word_length>1'"
FLIST="is_first_word word_onset positional lexical"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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

##############
# CASE STUDY #
##############
PATIENT='510'
DTYPE='micro'
FILTER='raw'
QTRAIN="'block in [1,3,5] and word_length>1'"
QTEST="'block in [1,3,5] and word_length>1'"
FLIST="is_first_word word_onset positional lexical"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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

##############
# CASE STUDY #
##############
PATIENT='513'
DTYPE='micro'
FILTER='raw'
QTRAIN="'block in [2,4,6] and word_length>1'"
QTEST="'block in [2,4,6] and word_length>1'"
FLIST="is_first_word word_onset positional semantics"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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

##############
# CASE STUDY #
##############
PATIENT='513'
DTYPE='micro'
FILTER='raw'
QTRAIN="'block in [1,3,5] and word_length>1'"
QTEST="'block in [1,3,5] and word_length>1'"
FLIST="is_first_word word_onset positional semantics"
filename_py="encoding_trf.py --patient "$PATIENT" --data-type "$DTYPE" --filter "$FILTER" --query-train "$QTRAIN" --query-test "$QTEST" --feature-list "$FLIST" --n-folds-inner "$CV_FOLDS_IN" --n-folds-outer "$CV_FOLDS_OUT" --ablation-method "$METHOD" --smooth "$SMOOTH" --decimate "$DECIMATE" --each-feature-value"

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

