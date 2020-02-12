import numpy as np
import h5py

h5f = h5py.File("20181113_Driver1_Trip1.hdf", "r")
keys = list(h5f['CAN'].keys())
row = h5f['CAN'][keys[0]].shape[0]
col = len(keys)
data = np.zeros((row,col))

for i in range(col):
	data[:,i] = h5f['CAN'][keys[i]][:,0]
	print(keys[i], np.mean(data[:,i]))

#np.savetxt('/home/mdzadik/CAN_data/hdf_dataset.csv',data,delimiter=",",fmt='%.4f')

# time = np.loadtxt('dataset_car1_mod.csv', delimiter=",", usecols=0, skiprows=1)
# data = np.loadtxt('dataset_car1_mod.csv', delimiter=",", usecols=(1,2,3,4,5,6), skiprows=1)

# mu = np.mean(data,axis=0)
# sigma = np.std(data,axis=0)
# label = np.zeros((data.shape[0],1))

# for i in range(data.shape[0]):
	# if((i>=250 and i<500) or (i>=3400 and i<3500)):
		# data[i,0] = np.random.normal(mu[0], sigma[0])
		# label[i,0] = 1
	# elif((i>=750 and i<1000) or (i>=3600 and i<3700)):
		# data[i,1] = np.random.normal(mu[1], sigma[1])
		# label[i,0] = 1
	# elif((i>=1250 and i<1500) or (i>=3800 and i<3900)):
		# data[i,2] = np.random.normal(mu[2], sigma[2])
		# label[i,0] = 1
	# elif((i>=1750 and i<2000) or (i>=4000 and i<4100)):
		# data[i,3] = np.random.normal(mu[3], sigma[3])
		# label[i,0] = 1
	# elif((i>=2250 and i<2500) or (i>=4200 and i<4300)):
		# data[i,4] = np.random.normal(mu[4], sigma[4])
		# label[i,0] = 1
	# elif((i>=2750 and i<3000) or (i>=4400 and i<4500)):
		# data[i,5] = np.random.normal(mu[5], sigma[5])
		# label[i,0] = 1

# np.savetxt('X_train2.txt', data[0:3200,:], delimiter=",", fmt='%.3f')
# np.savetxt('y_train2.txt', label[0:3200,:], fmt='%d')
# np.savetxt('X_test2.txt', data[3200:data.shape[0],:], delimiter=",", fmt='%.3f')
# np.savetxt('y_test2.txt', label[3200:data.shape[0],:], fmt='%d')
