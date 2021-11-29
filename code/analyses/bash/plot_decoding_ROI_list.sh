# Local(0) or Alambic (1)?
CLUSTER=1

#
CLASSIFIER='logistic'
SMOOTH=50
DECIMATE=50
TMIN=-0.1
TMAX=1

# TAKE ALL PATINETS
COMPARISONS="number embedding_vs_long dec_quest_len2"
DTYPE_FILTERS="micro_raw"
BLOCK_TRAINS="visual auditory"
BLOCK_TESTS="visual auditory"

# 
ROIs=""
HEMIS="lh rh"
for HEMI in $HEMIS; do
    for i in {6..47}; do
        ROIs=$ROIs"Brodmann."$i"-"$HEMI" "
    done
done

queue="Unicog_long"
walltime="72:00:00"

for BLOCK_TRAIN in $BLOCK_TRAINS; do
    for BLOCK_TEST in $BLOCK_TESTS; do
        for COMPARISON in $COMPARISONS; do
            for DTYPE_FILTER in $DTYPE_FILTERS; do
                for ROI in $ROIs; do
                        #echo $BLOCK_TRAIN $BLOCK_TEST $COMPARISON $DTYPE_FILTER $ROI $TMIN $TMAX $CLASSIFIER $SMOOTH $DECIMATE
                        path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"
                        filename_py="plot_decoding.py --comparison-name "$COMPARISON" --tmin "$TMIN" --tmax "$TMAX" --classifier "$CLASSIFIER" --ROIs "$ROI" --smooth "$SMOOTH" --data-type_filters "$DTYPE_FILTER" --decimate "$DECIMATE
                        output_log='logs/out_trf_'$ROI
                        error_log='logs/err_trf_'$ROI
                        job_name=$ROI

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
