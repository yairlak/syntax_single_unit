# SPIKE SORTING        
### Rename ncs files to csc.ncs format: copy the following to /ncs folder and run it (required for SPIKE SORTING with Combinato)
Code/Utils/spike_sorting/rename_channel_names_to_CSCs.py

### Run from Raw/micro/

### All the following steps should be run from the folder: /CSC_ncs

0. cd to path  
...`cd /Raw/micro/`, or with /mat, if recording files were transformed to mat files. 

1. (optional) generate overview plots for raw signal  
...`css-plot-rawsignal`  
...check new subfolder overview

2. extract spikes (not yet sorted) from all channels  
...`css-extract --files *.ncs > css_extract.log`  
...new subfolder CSC were genereted with data_CSC*.h5 files
... For BlackRock, there's a bash file for running it per each file (see, e.g., patient_504/Raw/micro/CSC_mat/css_extract.bash). It adds the --matfile flag. !!! --- Note that the mat files should contain a variable named 'sr' with the sampling rate (otherwise default will be used) --- !!! 

3. find artifacts  
...`css-find-concurrent`

4. remove artifacts  
...`css-mask-artifacts`

5. prepare for sorting  
...Use css-plot-extracted to create plots of the spikes after artifact removal. These plots display the different artifact types, and also visualize cumulative spike counts.  
...Next, prepare a job file (do_sort_pos.txt) by using css-overview-gui (Actions-->init from current folder and then  Actions->Save actions to file): this will generate the job file.  
...(!!--important--!!) remove duplications in this job file (not clear why it happens). Make sure you have #channels lines in this file, without duplications, before you continue.  
...Note that you can also prepare a job file for negative spikes (do_neg*.txt). For this, use css-overview-gui (toggle sort negative) to change the values in the corresponding channel rows. This will generate a second job file (do_sort_neg.txt). You would then need to repeat the step below TWICE, once for pos and once for neg.  

Run:  
`css-prepare-sorting --jobs do_sort_pos.txt`  
`css-prepare-sorting --jobs do_sort_neg.txt --neg`

6. Sorting  
...`css-cluster --jobs sort_pos_yl2.txt`  
...`css-cluster --jobs sort_neg_yl2.txt`

7. combine  
...`css-combine --jobs sort_pos_yl2.txt`  
...`css-combine --jobs sort_neg_yl2.txt`

8. (optional) generate sorting plots  
...`css-plot-sorted --label sort_pos_yl2`  
...`css-plot-sorted --label sort_neg_yl2`  

9. (optional) Use the GUIs to optimize results  
...`css-overview-gui`  
...enter the sorting label sort_pos_abc and initialize the folder (from the menu or by pressing Ctrl+I).

10. Manual fix:  
...`css-gui`

# GENERATE EPOCH FILES
- from Code/Main/micro, launch `generate_multichannel_spectrotemporal_epochs_micro.py`
- or, `Bash/generate_epochs/parallel_generate_epochsTRF_micro.sh`
- Same for macro electrodes or spikes. Replace in the above micro -> macro/spikes

### Generate ERSP and GAT figures
From Bash/plotting, launch:  
`parallel_plot_comparisons.py`

### Get probe names from all patients:
In Code/Main,   
`python get_probe_names.py --patient 479_11 --patient 482 --patient 487 --patient 493 --patient 502 --patient 504 --patient 505 --patient 510 --patient 513 --patient 515`

