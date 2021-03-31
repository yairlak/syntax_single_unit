import os, glob

ncs_files = glob.glob('CSC*.ncs')

for fn in sorted(ncs_files):
    bn = os.path.basename(fn)[0:-4]
    cmd = 'css-prepare-sorting --data ' + bn + '/data_' + bn + '.h5'
    os.system(cmd)
    cmd = 'css-prepare-sorting --neg --data ' + bn + '/data_' + bn + '.h5'
    os.system(cmd)


