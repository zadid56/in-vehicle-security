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
	if((i>=2000 and i<3500) or (i>=101000 and i<102000)):
		data[i,0] = data[i,0]+mid[0] if data[i,0]<mid[0] else data[i,0]-mid[0]
		#data[i,0] = np.random.uniform(low[0], high[0])
		label[i,0] = 1
	elif((i>=5000 and i<6500) or (i>=103000 and i<104000)):
		data[i,1] = data[i,1]+mid[1] if data[i,1]<mid[1] else data[i,1]-mid[1]
		label[i,0] = 1
	elif((i>=8000 and i<9500) or (i>=105000 and i<106000)):
		data[i,2] = data[i,2]+mid[2] if data[i,2]<mid[2] else data[i,2]-mid[2]
		label[i,0] = 1
	elif((i>=11000 and i<12500) or (i>=107000 and i<108000)):
		data[i,3] = data[i,3]+mid[3] if data[i,3]<mid[3] else data[i,3]-mid[3]
		label[i,0] = 1
	elif((i>=14000 and i<15500) or (i>=109000 and i<110000)):
		data[i,4] = data[i,4]+mid[4] if data[i,4]<mid[4] else data[i,4]-mid[4]
		label[i,0] = 1
	elif((i>=22000 and i<23500) or (i>=111000 and i<112000)):
		data[i,5] = data[i,5]+mid[5] if data[i,5]<mid[5] else data[i,5]-mid[5]
		label[i,0] = 1
	elif((i>=25000 and i<26500) or (i>=113000 and i<114000)):
		data[i,6] = data[i,6]+mid[6] if data[i,6]<mid[6] else data[i,6]-mid[6]
		label[i,0] = 1
	elif((i>=28000 and i<29500) or (i>=115000 and i<116000)):
		data[i,7] = data[i,7]+mid[7] if data[i,7]<mid[7] else data[i,7]-mid[7]
		label[i,0] = 1
	elif((i>=31000 and i<32500) or (i>=117000 and i<118000)):
		data[i,8] = data[i,8]+mid[8] if data[i,8]<mid[8] else data[i,8]-mid[8]
		label[i,0] = 1
	elif((i>=34000 and i<35500) or (i>=119000 and i<120000)):
		data[i,9] = data[i,9]+mid[9] if data[i,9]<mid[9] else data[i,9]-mid[9]
		label[i,0] = 1
	elif((i>=42000 and i<43500) or (i>=121000 and i<122000)):
		data[i,10] = data[i,10]+mid[10] if data[i,10]<mid[10] else data[i,10]-mid[10]
		label[i,0] = 1
	elif((i>=45000 and i<46500) or (i>=123000 and i<124000)):
		data[i,11] = data[i,11]+mid[11] if data[i,11]<mid[11] else data[i,11]-mid[11]
		label[i,0] = 1
	elif((i>=48000 and i<49500) or (i>=125000 and i<126000)):
		data[i,12] = data[i,12]+mid[12] if data[i,12]<mid[12] else data[i,12]-mid[12]
		label[i,0] = 1
	elif((i>=51000 and i<52500) or (i>=127000 and i<128000)):
		data[i,13] = data[i,13]+mid[13] if data[i,13]<mid[13] else data[i,13]-mid[13]
		label[i,0] = 1
	elif((i>=54000 and i<55500) or (i>=129000 and i<130000)):
		data[i,14] = data[i,14]+mid[14] if data[i,14]<mid[14] else data[i,14]-mid[14]
		label[i,0] = 1
	elif((i>=62000 and i<63500) or (i>=131000 and i<132000)):
		data[i,15] = data[i,15]+mid[15] if data[i,15]<mid[15] else data[i,15]-mid[15]
		label[i,0] = 1
	elif((i>=65000 and i<66500) or (i>=133000 and i<134000)):
		data[i,16] = data[i,16]+mid[16] if data[i,16]<mid[16] else data[i,16]-mid[16]
		label[i,0] = 1
	elif((i>=68000 and i<69500) or (i>=135000 and i<136000)):
		data[i,17] = data[i,17]+mid[17] if data[i,17]<mid[17] else data[i,17]-mid[17]
		label[i,0] = 1
	elif((i>=71000 and i<72500) or (i>=137000 and i<138000)):
		data[i,18] = data[i,18]+mid[18] if data[i,18]<mid[18] else data[i,18]-mid[18]
		label[i,0] = 1
	elif((i>=74000 and i<75500) or (i>=139000 and i<140000)):
		data[i,19] = data[i,19]+mid[19] if data[i,19]<mid[19] else data[i,19]-mid[19]
		label[i,0] = 1
	elif((i>=82000 and i<83500) or (i>=141000 and i<142000)):
		data[i,20] = data[i,20]+mid[20] if data[i,20]<mid[20] else data[i,20]-mid[20]
		label[i,0] = 1
	elif((i>=85000 and i<86500) or (i>=143000 and i<144000)):
		data[i,21] = data[i,21]+mid[21] if data[i,21]<mid[21] else data[i,21]-mid[21]
		label[i,0] = 1
	elif((i>=88000 and i<89500) or (i>=145000 and i<146000)):
		data[i,22] = data[i,22]+mid[22] if data[i,22]<mid[22] else data[i,22]-mid[22]
		label[i,0] = 1
	elif((i>=91000 and i<92500) or (i>=147000 and i<148000)):
		data[i,23] = data[i,23]+mid[23] if data[i,23]<mid[23] else data[i,23]-mid[23]
		label[i,0] = 1

np.savetxt('X_train3.txt', data[0:95000,:], delimiter=",", fmt='%.4f')
np.savetxt('y_train3.txt', label[0:95000,0], fmt='%d')
np.savetxt('X_test3.txt', data[95000:data.shape[0],:], delimiter=",", fmt='%.4f')
np.savetxt('y_test3.txt', label[95000:data.shape[0],0], fmt='%d')
