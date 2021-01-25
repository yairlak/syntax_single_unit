# Print responsive and amodoal channels for all cases
#data_type='micro'
#level='sentence_offset'
filter='gaussian-kernel'



python plot_amodal_channels.py --patient 479_11 479_25 482 487 493 502 504 505 510 513 515 --data-type macro micro spike --filter gaussian-kernel --level sentence_onset sentence_offset


exit 1
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


for data_type in 'micro' 'macro' 'spike'; do
    for level in 'sentence_onset' 'sentence_offset';do
        for patient in '479_11' '479_25' '482' '487' '493' '502' '504' '505' '510' '513' '515'; do
			output_log='logs/out_plot_amodal'
			error_log='logs/err_plot_amodal'
			job_name='plot_amodal'
                        CMD='python /neurospin/unicog/protocols/intracranial/Syntax_with_Fried/Code/Main/plot_amodal_channels.py --patient '$patient' --filter '$filter' --level '$level' --data-type '$data_type
			if [ $CLUSTER -eq 1 ]
			then
				echo $CMD | qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log
			else
				echo $CMD # ' 1>'$output_log' 2>'$error_log' &'
			fi
        done
    done
done
