import os, glob
ncs_files = glob.glob('CSC*/data_CSC*.h5')
for fn in sorted(ncs_files):
    print(fn)
    bn = os.path.basename(fn)
    cmd = 'css-plot-sorted --datafile %s --label sort_pos_yl2' % bn
    os.system(cmd)
    cmd = 'css-plot-sorted --datafile %s --label sort_neg_yl2' % bn
    os.system(cmd)
