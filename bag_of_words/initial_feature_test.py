import numpy as np
#import pandas as pd
import mne
from mne import io
import eeghdf
from eeghdf import convert
import mne_features
from mne_features import univariate
import matplotlib.pyplot as plt


def feature_extraction(data, sfreq):
	#What features do I want???
	#Kurtosis
	#line-length
	#entropy

	#Perform feature extraction for 2 second segments
	kurt = np.zeros((data.shape[0],data.shape[1]//2//sfreq))
	s = np.zeros((data.shape[0],data.shape[1]//2//sfreq))
	H = np.zeros((data.shape[0],data.shape[1]//2//sfreq))


	segments = np.zeros((data.shape[0],data.shape[1]//2//sfreq,sfreq*2))
	for n in range(0,data.shape[1]//2//sfreq):
		cur_data = data[:,n*sfreq*2:(n+1)*sfreq*2]
		segments[:,n,:] = cur_data
		kurt[:,n] = univariate.compute_kurtosis(cur_data)
		s[:,n] = univariate.compute_line_length(cur_data)
		H[:,n] = univariate.compute_spect_entropy(sfreq, cur_data, psd_method='welch')

	#Average across channels
	kurt_avg = np.mean(kurt,axis=0)
	s_avg = np.mean(s,axis=0)*np.power(10,6)
	H_avg = np.mean(H,axis=0)

	return kurt_avg, s_avg, H_avg

#Abnormal Recording
raw_fname = 'CA7551E5_1-3+.eeghdf'
hf = eeghdf.Eeghdf(raw_fname)
raw, info, channels  = convert.hdf2mne(hf)



#Extract actual data
data, times = raw[:]
sfreq = int(np.rint(raw.info['sfreq']))

ab_kurt, ab_length, ab_ent = feature_extraction(data, sfreq)

print('Abnormal Done')

#Normal Recording
raw_fname = 'CA84303Q_1-1+.eeghdf'
hf = eeghdf.Eeghdf(raw_fname)
raw, info, channels  = convert.hdf2mne(hf)

#Extract actual data
data, times = raw[:]
sfreq = int(np.rint(raw.info['sfreq']))
print(times[-1])
print(sfreq)

nor_kurt, nor_length, nor_ent = feature_extraction(data, sfreq)


print('Normal Done')

#Seizure Recording
raw_fname = 'CA3465RA_1-1+.eeghdf'
hf = eeghdf.Eeghdf(raw_fname)
raw, info, channels  = convert.hdf2mne(hf)

#Extract actual data
data, times = raw[:]
sfreq = int(np.rint(raw.info['sfreq']))
print(times[-1])
print(sfreq)


sz_kurt, sz_length, sz_ent = feature_extraction(data, sfreq)

print('Seizure Done')

#Spikes Recordingprint(times[-1])
print(sfreq)

raw_fname = 'CA34675Q_1-1+.eeghdf'
hf = eeghdf.Eeghdf(raw_fname)
raw, info, channels  = convert.hdf2mne(hf)

#Extract actual data
data, times = raw[:]
sfreq = int(np.rint(raw.info['sfreq']))
print(times[-1])
print(sfreq)


spk_kurt, spk_length, spk_ent = feature_extraction(data, sfreq)

print('Spikes Done')


plt.figure(1)
time = np.arange(ab_kurt.size)*2
plt.plot(time,ab_kurt)
plt.plot(time,ab_length)
plt.plot(time,ab_ent)
plt.title('Features of Abnormal EEG')
plt.ylabel('Different for Everything')
plt.xlabel('Time (s)')
plt.legend(['Kurtosis', 'Line Length', 'Spectral Entropy'])

plt.figure(2)
time = np.arange(nor_kurt.size)*2
plt.plot(time,nor_kurt)
plt.plot(time,nor_length)
plt.plot(time,nor_ent)
plt.title('Features of Normal EEG')
plt.ylabel('Different for Everything')
plt.xlabel('Time (s)')
plt.legend(['Kurtosis', 'Line Length', 'Spectral Entropy'])

plt.figure(3)
time = np.arange(sz_kurt.size)*2
plt.plot(time,sz_kurt)
plt.plot(time,sz_length)
plt.plot(time,sz_ent)
plt.title('Features of Seizure EEG')
plt.ylabel('Different for Everything')
plt.xlabel('Time (s)')
plt.legend(['Kurtosis', 'Line Length', 'Spectral Entropy'])

plt.figure(4)
time= np.arange(spk_kurt.size)*2
plt.plot(time,spk_kurt)
plt.plot(time,spk_length)
plt.plot(time,spk_ent)
plt.title('Features of Spiking EEG')
plt.ylabel('Different for Everything')
plt.xlabel('Time (s)')
plt.legend(['Kurtosis', 'Line Length', 'Spectral Entropy'])

plt.show()

