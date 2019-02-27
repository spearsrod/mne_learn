import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import mne
from mne import io
from mne.datasets import sample
from mne.filter import filter_data
from mne import decoding
from mne.connectivity import spectral_connectivity
from mayavi import mlab 


# Load BECTS EEG
raw_fname = 'bects_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=True)
#print(raw.info)

# Sampling Variables

#Sampling frequency (Hz)
fs = raw.info['sfreq']

f_low = 0.5
f_high = 40
delta = [0.5, 3.5]
theta = [4, 8]
alpha = [8.5, 13]
beta = [14, 30]


#Processing Steps

#1. Re-referencing to average
raw.set_eeg_reference('average')

#2. Z-score
#raw.apply_baseline(mode='zscore')
#raw.Scalar(scalings='mean')

#3. Band Pass (0.5 Hz to 40 Hz)

raw.filter(f_low, f_high)

#Stationary Partitions (2-second)

#Random selection of N 2 second partitions (N = 5)

picks = mne.pick_types(raw.info,meg=False, eeg=True, stim=False, eog=False)

intervals = np.random.randint(1,raw.n_times - 2*fs,size=5)

event_list = []
for idx, n in enumerate(intervals):
	event_list.append([int(n), 0, idx+1])

print("Indicies of Randomly Selected 2-s Interval Times")
print(event_list)

#See 
events = np.array(event_list)
epochs = mne.Epochs(raw, events, tmin=0.0, tmax=2.0, picks=picks)


#Determine Spectral Connectivity in Each Band using Weighted Phase Locking
con_d, freqs_d, times_d, n_epochs_d, n_tapers_d = spectral_connectivity(epochs, method='wpli', mode='multitaper', sfreq=fs,
    fmin=delta[0], fmax=delta[1], faverage=True, n_jobs=1)
con_th, freqs_th, times_th, n_epochs_th, n_tapers_th = spectral_connectivity(epochs, method='wpli', mode='multitaper', sfreq=fs,
    fmin=theta[0], fmax=theta[1], faverage=True, n_jobs=1)
con_a, freqs_a, times_a, n_epochs_a, n_tapers_a = spectral_connectivity(epochs, method='wpli', mode='multitaper', sfreq=fs,
    fmin=alpha[0], fmax=alpha[1], faverage=True, n_jobs=1)
con_b, freqs_b, times_b, n_epochs_b, n_tapers_b = spectral_connectivity(epochs, method='wpli', mode='multitaper', sfreq=fs,
    fmin=beta[0], fmax=beta[1], faverage=True, n_jobs=1)



#Select Connectivity to Plot
con = con_b

#Include plotting method from connectivity.py
layout = mne.find_layout(epochs.info, 'eeg')  # use full layout

# get rid of extra dimension
con = con[:, :, 0] # make it 2D example: shape  is (25,25)

# remove EOG Pg1 and Pg2

ch_names = epochs.ch_names
idx = [ch_names.index(name) for name in ch_names if not name[:2]=='Pg' ]
con = con[idx][:, idx]



mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

# Plot the sensor locations
#print(picks)
sens_loc = [raw.info['chs'][picks[i]]['loc'][:3] for i in idx]
sens_loc = np.array(sens_loc)

pts = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                    color=(1, 1, 1), opacity=1, scale_factor=0.005)


n_con = 23   # show up to 10 connections maybe choose 23
min_dist = 0.05  # exclude sensors that are less than 5cm apart
threshold = np.sort(con, axis=None)[-n_con]
ii, jj = np.where(con >= threshold)


# Remove close connections
con_nodes = list()
con_val = list()
for i, j in zip(ii, jj):
    if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
        con_nodes.append((i, j))
        #print('include:', con[i,j])
        con_val.append(con[i, j])

con_val = np.array(con_val)

# Show the connections as tubes between sensors
vmax = np.max(con_val)
vmin = np.min(con_val)
for val, nodes in zip(con_val, con_nodes):
    x1, y1, z1 = sens_loc[nodes[0]]
    x2, y2, z2 = sens_loc[nodes[1]]
    points = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                         vmin=vmin, vmax=vmax, tube_radius=0.001,
                         colormap='RdBu')
    points.module_manager.scalar_lut_manager.reverse_lut = True


mlab.scalarbar(points, title='Beta Band Phase Lag Index (PLI)', nb_labels=4)

# Add the sensor names for the connections shown
nodes_shown = list(set([n[0] for n in con_nodes] +
                       [n[1] for n in con_nodes]))

print(nodes_shown)
for node in nodes_shown:
    x, y, z = sens_loc[node]
    mlab.text3d(x, y, z, raw.ch_names[picks[node]], scale=0.005,
                color=(0, 0, 0))

view = (-88.7, 40.8, 0.76, np.array([-3.9e-4, -8.5e-3, -1e-2]))
mlab.view(*view)

#seed_ch_num = int(np.where(indices[1] == seed)[0])
plt.imshow(con)
plt.title('Beta Band Connectivity Matrix')
plt.show()




##Additional Steps
# Zscore across single channel and all channels
# Calculate histogram for N~50 5 epoch samples for each channel relative to some seed channel



