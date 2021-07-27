# Imports
from brpylib import NsxFile, brpylib_ver
import matplotlib.pyplot as plt

# Version control
brpylib_ver_req = "1.2.1"
if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
    raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

# Inits
datafile = '../../../Data/UCLA/patient_539/Raw/macro/'

# Open file and extract headers
nsx_file = NsxFile(datafile)

# Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
cont_data = nsx_file.getdata()

# Close the nsx file now that all data is out
nsx_file.close()

# Plot the data channel
plot_chan = 1
ch_idx  = cont_data['elec_ids'].index(plot_chan)
hdr_idx = cont_data['ExtendedHeaderIndices'][ch_idx]
t       = cont_data['start_time_s'] + arange(cont_data['data'].shape[1]) / cont_data['samp_per_s']

plt.plot(t, cont_data['data'][ch_idx])
plt.axis([t[0], t[-1], min(cont_data['data'][ch_idx]), max(cont_data['data'][ch_idx])])
plt.locator_params(axis='y', nbins=20)
plt.xlabel('Time (s)')

