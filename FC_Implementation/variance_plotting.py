import matplotlib.pyplot as plt
import numpy as np


d_con_all = np.load('delta_con.npy')
th_con_all = np.load('theta_con.npy')
a_con_all = np.load('alpha_con.npy')
b_con_all = np.load('beta_con.npy')

for expr in range(0,50):
	d_con_all[expr,:,:] = np.tril(d_con_all[expr,:,:]) + np.triu(d_con_all[expr,:,:].T, 1)
	th_con_all[expr,:,:] = np.tril(th_con_all[expr,:,:]) + np.triu(th_con_all[expr,:,:].T, 1)
	a_con_all[expr,:,:] = np.tril(a_con_all[expr,:,:]) + np.triu(a_con_all[expr,:,:].T, 1)
	b_con_all[expr,:,:] = np.tril(b_con_all[expr,:,:]) + np.triu(b_con_all[expr,:,:].T, 1)

#Choose connection to plot
con = 2

test = a_con_all[:,con,:].shape
print(test)
mean_con = np.mean(a_con_all[:,con,:],axis=0)
std_con = np.std(a_con_all[:,con,:],axis=0)
upper_con = mean_con + std_con
lower_con = mean_con - std_con
print(mean_con)

plt.figure(1)
for expr in range(0,50):
	connection = a_con_all[expr,con,:]
	plt.plot(np.delete(range(0,23),con),np.delete(connection,con), '*')
plt.plot(np.delete(range(0,23),con), np.delete(mean_con,con))
plt.fill_between(np.delete(range(0,23),con), np.delete(upper_con,con), np.delete(lower_con,con),alpha=0.5)
plt.xlabel('EEG Index')
plt.ylabel('Covariance')
plt.title('Mean + Std of 50 connectivity trials')
plt.show()

