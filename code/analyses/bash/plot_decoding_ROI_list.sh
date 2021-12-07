# Local(0) or Alambic (1)?
CLUSTER=1

#
CLASSIFIER='logistic'
SMOOTH=50
DECIMATE=50
TMIN=-0.2
TMAX=1.2
FROM_PKL="" # " --from-pkl" or ""

# TAKE ALL PATINETS
COMPARISONS="number embedding_vs_long dec_quest_len2 pos_simple word_string_first"
#COMPARISONS="pos_simple word_string_first"
DTYPE_FILTERS="micro_raw spike_raw micro_high-gamma"
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

queue="Nspin_long"
walltime="72:00:00"

for BLOCK_TRAIN in $BLOCK_TRAINS; do
    for BLOCK_TEST in $BLOCK_TESTS; do
        for COMPARISON in $COMPARISONS; do
            for DTYPE_FILTER in $DTYPE_FILTERS; do
                for ROI in $ROIs; do
                        #echo $BLOCK_TRAIN $BLOCK_TEST $COMPARISON $DTYPE_FILTER $ROI $TMIN $TMAX $CLASSIFIER $SMOOTH $DECIMATE
                        path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"
                        filename_py="plot_decoding.py --comparison-name "$COMPARISON" --tmin "$TMIN" --tmax "$TMAX" --classifier "$CLASSIFIER" --ROIs "$ROI" --smooth "$SMOOTH" --data-type_filters "$DTYPE_FILTER" --decimate "$DECIMATE" --block-train "$BLOCK_TRAIN" --block-test "$BLOCK_TEST$FROM_PKL
			#" --min-trials 18 --fixed-constraint 'word_position==1'"
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

# HACK
# ONE MORE TIME FOR BOTH MICRO AND SPIKE

#for BLOCK_TRAIN in $BLOCK_TRAINS; do
#    for BLOCK_TEST in $BLOCK_TESTS; do
#        for COMPARISON in $COMPARISONS; do
#                for ROI in $ROIs; do
#                        #echo $BLOCK_TRAIN $BLOCK_TEST $COMPARISON $DTYPE_FILTER $ROI $TMIN $TMAX $CLASSIFIER $SMOOTH $DECIMATE
#                        path2script="/neurospin/unicog/protocols/intracranial/syntax_single_unit/code/analyses/"
#                        filename_py="decoding.py --comparison-name "$COMPARISON" --tmin "$TMIN" --tmax "$TMAX" --classifier "$CLASSIFIER" --ROIs "$ROI" --smooth "$SMOOTH" --data-type_filters micro_raw spike_raw --decimate "$DECIMATE" --block-train "$BLOCK_TRAIN" --block-test "$BLOCK_TEST
#                        output_log='logs/out_trf_'$ROI
#                        error_log='logs/err_trf_'$ROI
#                        job_name=$ROI
#
#                        CMD="python3 $path2script$filename_py"
#
#                        if [ $CLUSTER -eq 1 ]
#                        then
#                            echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
#                        else
#                            echo $CMD
#                            #' 1>'$output_log' 2>'$error_log 
#                        fi
#                done
#        done
#    done
#done
