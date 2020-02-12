import numpy as np

#time = np.loadtxt('dataset_car1_mod.csv', delimiter=",", usecols=0, skiprows=1)
#data = np.loadtxt('dataset_car1_mod.csv', delimiter=",", usecols=(1,2,3,4,5,6), skiprows=1)
data = np.loadtxt('hdf_dataset.csv', delimiter=",")

mu = np.mean(data,axis=0)
sigma = np.std(data,axis=0)
low = np.min(data,axis=0)
high = np.max(data,axis=0)
mid = (low+high)/2
label = np.zeros((data.shape[0],1))

for i in range(data.shape[0]):
	if((i>=2000 and i<3000) or (i>=101000 and i<101500)):
		data[i,0] = np.random.uniform(low[0], high[0])
		label[i,0] = 1
	elif((i>=5000 and i<3000) or (i>=103000 and i<103500)):
		data[i,1] = np.random.uniform(low[1], high[1])
		label[i,0] = 1
	elif((i>=8000 and i<9000) or (i>=105000 and i<105500)):
		data[i,2] = np.random.uniform(low[2], high[2])
		label[i,0] = 1
	elif((i>=11000 and i<12000) or (i>=107000 and i<107500)):
		data[i,3] = np.random.uniform(low[3], high[3])
		label[i,0] = 1
	elif((i>=14000 and i<15000) or (i>=109000 and i<109500)):
		data[i,4] = np.random.uniform(low[4], high[4])
		label[i,0] = 1
	elif((i>=22000 and i<23000) or (i>=111000 and i<111500)):
		data[i,5] = np.random.uniform(low[5], high[5])
		label[i,0] = 1
	elif((i>=25000 and i<26000) or (i>=113000 and i<113500)):
		data[i,6] = np.random.uniform(low[6], high[6])
		label[i,0] = 1
	elif((i>=28000 and i<29000) or (i>=115000 and i<115500)):
		data[i,7] = np.random.uniform(low[7], high[7])
		label[i,0] = 1
	elif((i>=31000 and i<32000) or (i>=117000 and i<117500)):
		data[i,8] = np.random.uniform(low[8], high[8])
		label[i,0] = 1
	elif((i>=34000 and i<35000) or (i>=119000 and i<119500)):
		data[i,9] = np.random.uniform(low[9], high[9])
		label[i,0] = 1
	elif((i>=42000 and i<43000) or (i>=121000 and i<121500)):
		data[i,10] = np.random.uniform(low[10], high[10])
		label[i,0] = 1
	elif((i>=45000 and i<46000) or (i>=123000 and i<123500)):
		data[i,11] = np.random.uniform(low[11], high[11])
		label[i,0] = 1
	elif((i>=48000 and i<49000) or (i>=125000 and i<125500)):
		data[i,12] = np.random.uniform(low[12], high[12])
		label[i,0] = 1
	elif((i>=51000 and i<52000) or (i>=127000 and i<127500)):
		data[i,13] = np.random.uniform(low[13], high[13])
		label[i,0] = 1
	elif((i>=54000 and i<55000) or (i>=129000 and i<129500)):
		data[i,14] = np.random.uniform(low[14], high[14])
		label[i,0] = 1
	elif((i>=62000 and i<63000) or (i>=131000 and i<131500)):
		data[i,15] = np.random.uniform(low[15], high[15])
		label[i,0] = 1
	elif((i>=65000 and i<66000) or (i>=133000 and i<133500)):
		data[i,16] = np.random.uniform(low[16], high[16])
		label[i,0] = 1
	elif((i>=68000 and i<69000) or (i>=135000 and i<135500)):
		data[i,17] = np.random.uniform(low[17], high[17])
		label[i,0] = 1
	elif((i>=71000 and i<72000) or (i>=137000 and i<137500)):
		data[i,18] = np.random.uniform(low[18], high[18])
		label[i,0] = 1
	elif((i>=74000 and i<75000) or (i>=139000 and i<139500)):
		data[i,19] = np.random.uniform(low[19], high[19])
		label[i,0] = 1
	elif((i>=82000 and i<83000) or (i>=141000 and i<141500)):
		data[i,20] = np.random.uniform(low[20], high[20])
		label[i,0] = 1
	elif((i>=85000 and i<86000) or (i>=143000 and i<143500)):
		data[i,21] = np.random.uniform(low[21], high[21])
		label[i,0] = 1
	elif((i>=88000 and i<89000) or (i>=145000 and i<145500)):
		data[i,22] = np.random.uniform(low[22], high[22])
		label[i,0] = 1
	elif((i>=91000 and i<92000) or (i>=147000 and i<147500)):
		data[i,23] = np.random.uniform(low[23], high[23])
		label[i,0] = 1

np.savetxt('X_train3.txt', data[0:95000,:], delimiter=",", fmt='%.4f')
np.savetxt('y_train3.txt', label[0:95000,0], fmt='%d')
np.savetxt('X_test3.txt', data[95000:data.shape[0],:], delimiter=",", fmt='%.4f')
np.savetxt('y_test3.txt', label[95000:data.shape[0],0], fmt='%d')
