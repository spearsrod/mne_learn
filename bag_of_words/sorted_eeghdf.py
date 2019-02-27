import numpy as np
import pandas as pd



filename = 'all_reports_impress_blanked-2019-02-23.csv'
df = pd.read_csv(filename, delimiter=',',dtype=str)

stan = df.loc[df['database_source'] == 'STANFORD_NK']
stan = df.loc[df['data_source'] == 'CLARITY_SHC']
stan['eeghdf_file'] = stan['edf_file_name'].str.slice(start=0,stop=-2) + 'eghdf'
print(stan['eeghdf_file'].size)

for idx in range(0,stan['eeghdf_file'].size):


#print(stan['eeghdf_file'])

# #normal
# CA84303Q_1-1+.edf

# #seizure
# CA3465RA_1-1+.edf

# #spikes
# CA34675Q_1-1+.edf


