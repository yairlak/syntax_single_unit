import os, glob

ncs_files = glob.glob('CSC*.ncs')

for fn in sorted(ncs_files):
    bn = os.path.basename(fn)[0:-4]
#    if int(bn[3:]) in [8, 9] + list(range(80, 89)):
    cmd = 'css-simple-clustering --datafile ' + bn + '/data_' + bn + '.h5'
    os.system(cmd)
    cmd = 'css-simple-clustering --datafile ' + bn + '/data_' + bn + '.h5 --neg'
    os.system(cmd)


