import numpy as np
probe_names = ['LMI', 'LEC', 'LA', 'LPHG', 'RAH', 'REC', 'LOF', 'LIF', 'LAI', 'LAH']

f = open('channel_numbers_to_names.txt', 'w')
f.write('%i\t%s\n'%(0, 'MICROPHONE'))
for p, probe_name in enumerate(probe_names):
    group_of_probes = int(np.floor(p/4)) # grouped into four GA1, GA2, GA3, GA4; GB1, GB2,...
    prefix_letter = ['A', 'B', 'C'][group_of_probes]
    prefix_index = str(p%4+1)
    for i in range(1, 9):
        ch_name = 'G'+prefix_letter+prefix_index+'-'+probe_name+str(i)+'.BKR'
        f.write('%i\t%s\n'%(i, ch_name))
