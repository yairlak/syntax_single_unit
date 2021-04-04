# Run this script from the CSC_mat folder
import os, glob
from scipy import io

CSC_files = glob.glob(os.path.join('CSC*.mat'))

channel_numbers_to_names = []
for fn in sorted(CSC_files):
    base_name = os.path.basename(fn)[0:-4]
    CSC_number = int(base_name[3::])


    CSC = io.loadmat(fn)
    if 'file_name' in CSC.keys():
        channel_name = CSC['file_name'][0]
    else:
        channel_name = base_name
        print('No channel name for %s' % fn)
    channel_numbers_to_names.append((CSC_number, channel_name))
    print('processed: %s, %s, %s' % (CSC_number, base_name, channel_name))

channel_numbers_to_names.sort(key=lambda sublist: sublist[0])
with open('channel_numbers_to_names.txt', 'w') as file:
    file.writelines('\t'.join(map(str, i)) + '\n' for i in channel_numbers_to_names)

