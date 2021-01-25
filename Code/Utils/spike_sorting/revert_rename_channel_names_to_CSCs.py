import os
with open('../../ChannelsCSC/micro/channel_numbers_to_names.txt') as f:
    nums2names = f.readlines()
nums2names = [(l.strip('\n').split('\t')[0], l.strip('\n').split('\t')[1]) for l in nums2names]

for (num, name) in nums2names:
    if int(num) > 0: # skip the mic channel
        os.rename('CSC'+num+'.ncs', name)
        print(num, name)
