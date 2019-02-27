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
print(raw.info['chs'])

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

intervals = np.random.randint(1,raw.n_times - 2*fs,size=250)
print('random_invervals')
print(intervals)

event_list = []
count = 0
run_count = -1

for idx, n in enumerate(intervals):
    if(np.mod(count,5)==0):
        event_list.append([])
        run_count += 1
    event_list[run_count].append([int(n), 0, idx+1])
    count += 1

print("Indicies of Randomly Selected 2-s Interval Times")
print(event_list)

#Choose channel to look at

con_d_all = np.zeros((50,23,23))
con_th_all = np.zeros((50,23,23))
con_a_all = np.zeros((50,23,23))
con_b_all = np.zeros((50,23,23))


for experiment in range(0,50):

    #See 
    events = np.array(event_list[experiment])
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
    
    con_d_all[experiment,:,:] = con_d[:,:,0]
    con_th_all[experiment,:,:] = con_th[:,:,0]
    con_a_all[experiment,:,:] = con_a[:,:,0]
    con_b_all[experiment,:,:] = con_b[:,:,0]




np.save('delta_con.npy',con_d_all)
np.save('theta_con.npy',con_th_all)
np.save('alpha_con.npy',con_a_all)
np.save('beta_con.npy',con_b_all)