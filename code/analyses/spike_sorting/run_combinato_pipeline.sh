#!/bin/bash -x

# BEFORE RUNNING THE SCRIPT ACTIVATE THE COMBINATO ENV:
# conda activate combinato
# !!!!!!!!!!!!!!!!!!!!!!

PATIENT='patient_491'

PATH_MAIN='/neurospin/unicog/protocols/intracranial/syntax_single_unit/'

mkdir $PATH_MAIN'Data/UCLA/'$PATIENT'/Raw/micro/CSC_ncs'

cp rename_channel_names_to_CSCs.py $PATH_MAIN'Data/UCLA/'$PATIENT'/Raw/micro/CSC_ncs'
python $PATH_MAIN'Data/UCLA/'$PATIENT'/Raw/micro/CSC_ncs/rename_channel_names_to_CSCs.py'

cd $PATH_MAIN'Data/UCLA/'$PATIENT'/Raw/micro/CSC_ncs'
# css-plot-rawsignal

# Extract spikes (not yet sorted) from all channels  
css-extract --files *.ncs > $PATIENT'.log'
# new subfolder CSC were genereted with data_CSC*.h5 files
# For BlackRock, there's a bash file for running it per each file (see, e.g., patient_504/Raw/micro/CSC_mat/css_extract.bash). It adds the --matfile flag. !!! --- Note that the mat files should contain a variable named 'sr' with the sampling rate (otherwise default will be used) --- !!! 

# Find artifacts  
css-find-concurrent >> $PATIENT'.log'

# Remove artifacts  
css-mask-artifacts >> $PATIENT'.log'

# Prepare for sorting  
# Prepare a job file (do_sort_pos.txt).  
# Consider preparing a job file also for negative spikes (do_sort_neg.txt).  

css-prepare-sorting --jobs do_sort_pos.txt >> $PATIENT'.log'
# css-prepare-sorting --jobs do_sort_neg.txt --neg

# Sorting  
css-cluster --jobs sort_pos_yl2.txt >> $PATIENT'.log'
# css-cluster --jobs sort_neg_yl2.txt

# combine  
css-combine --jobs sort_pos_yl2.txt >> $PATIENT'.log'
# css-combine --jobs sort_neg_yl2.txt

# css-plot-sorted --label sort_pos_yl2
# css-overview-gui
# css-gui
