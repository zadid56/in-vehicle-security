import numpy as np

data = np.loadtxt('hdf_dataset_3.csv', delimiter=",")

mu = np.mean(data,axis=0)
sigma = np.std(data,axis=0)
low = np.min(data,axis=0)
high = np.max(data,axis=0)
mid = (low+high)/2
label = np.zeros((data.shape[0],1))

for i in range(data.shape[0]):
	if((i>=2000 and i<4000) or (i>=102000 and i<103000)):
		data[i,0] = high[0]+low[0]-data[i,0]
		#data[i,0] = np.random.uniform(low[0], high[0])
		label[i,0] = 1
	elif((i>=6000 and i<8000) or (i>=104000 and i<105000)):
		data[i,1] = high[1]+low[1]-data[i,1]
		label[i,0] = 1
	elif((i>=12000 and i<14000) or (i>=106000 and i<107000)):
		data[i,2] = high[2]+low[2]-data[i,2]
		label[i,0] = 1
	elif((i>=16000 and i<18000) or (i>=108000 and i<109000)):
		data[i,3] = high[3]+low[3]-data[i,3]
		label[i,0] = 1
	elif((i>=22000 and i<24000) or (i>=112000 and i<113000)):
		data[i,4] = high[4]+low[4]-data[i,4]
		label[i,0] = 1
	elif((i>=26000 and i<28000) or (i>=114000 and i<115000)):
		data[i,5] = high[5]+low[5]-data[i,5]
		label[i,0] = 1
	elif((i>=32000 and i<34000) or (i>=116000 and i<117000)):
		data[i,6] = high[6]+low[6]-data[i,6]
		label[i,0] = 1
	elif((i>=36000 and i<38000) or (i>=118000 and i<119000)):
		data[i,7] = high[7]+low[7]-data[i,7]
		label[i,0] = 1
	elif((i>=42000 and i<44000) or (i>=122000 and i<123000)):
		data[i,8] = high[8]+low[8]-data[i,8]
		label[i,0] = 1
	elif((i>=46000 and i<48000) or (i>=124000 and i<125000)):
		data[i,9] = high[9]+low[9]-data[i,9]
		label[i,0] = 1
	elif((i>=52000 and i<54000) or (i>=126000 and i<127000)):
		data[i,10] = high[10]+low[10]-data[i,10]
		label[i,0] = 1
	elif((i>=56000 and i<58000) or (i>=128000 and i<129000)):
		data[i,11] = high[11]+low[11]-data[i,11]
		label[i,0] = 1
	elif((i>=62000 and i<64000) or (i>=132000 and i<133000)):
		data[i,12] = high[12]+low[12]-data[i,12]
		label[i,0] = 1
	elif((i>=66000 and i<68000) or (i>=134000 and i<135000)):
		data[i,13] = high[13]+low[13]-data[i,13]
		label[i,0] = 1
	elif((i>=72000 and i<74000) or (i>=136000 and i<137000)):
		data[i,14] = high[14]+low[14]-data[i,14]
		label[i,0] = 1
	elif((i>=76000 and i<78000) or (i>=138000 and i<139000)):
		data[i,15] = high[15]+low[15]-data[i,15]
		label[i,0] = 1
	elif((i>=82000 and i<84000) or (i>=142000 and i<143000)):
		data[i,16] = high[16]+low[16]-data[i,16]
		label[i,0] = 1
	elif((i>=86000 and i<88000) or (i>=144000 and i<145000)):
		data[i,17] = high[17]+low[17]-data[i,17]
		label[i,0] = 1
	elif((i>=92000 and i<94000) or (i>=146000 and i<147000)):
		data[i,18] = high[18]+low[18]-data[i,18]
		label[i,0] = 1

np.savetxt('X_train3.txt', data[0:95000,:], delimiter=",", fmt='%.4f')
np.savetxt('y_train3.txt', label[0:95000,0], fmt='%d')
np.savetxt('X_test3.txt', data[95000:data.shape[0],:], delimiter=",", fmt='%.4f')
np.savetxt('y_test3.txt', label[95000:data.shape[0],0], fmt='%d')
