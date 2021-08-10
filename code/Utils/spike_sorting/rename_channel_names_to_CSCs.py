# Run from Raw/micro/ncs/
import os, shutil
with open('../CSC_mat/channel_numbers_to_names.txt', 'r') as f:
    nums2names = f.readlines()
nums2names = [(l.strip('\n').split('\t')[0], l.strip('\n').split('\t')[1]) for l in nums2names]

for (num, name) in nums2names:
    if int(num) > 0: # skip the mic channel
        shutil.copyfile(name, os.path.join('..', 'spike', 'CSC'+num+'.ncs'))
        print(num, name)
