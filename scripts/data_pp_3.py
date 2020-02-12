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
		data[i,0] = data[i,0]+mid[0] if data[i,0]<mid[0] else data[i,0]-mid[0]
		#data[i,0] = np.random.uniform(low[0], high[0])
		label[i,0] = 1
	elif((i>=6000 and i<8000) or (i>=104000 and i<105000)):
		data[i,1] = data[i,1]+mid[1] if data[i,1]<mid[1] else data[i,1]-mid[1]
		label[i,0] = 1
	elif((i>=12000 and i<14000) or (i>=106000 and i<107000)):
		data[i,2] = data[i,2]+mid[2] if data[i,2]<mid[2] else data[i,2]-mid[2]
		label[i,0] = 1
	elif((i>=16000 and i<18000) or (i>=108000 and i<109000)):
		data[i,3] = data[i,3]+mid[3] if data[i,3]<mid[3] else data[i,3]-mid[3]
		label[i,0] = 1
	elif((i>=22000 and i<24000) or (i>=112000 and i<113000)):
		data[i,4] = data[i,4]+mid[4] if data[i,4]<mid[4] else data[i,4]-mid[4]
		label[i,0] = 1
	elif((i>=26000 and i<28000) or (i>=114000 and i<115000)):
		data[i,5] = data[i,5]+mid[5] if data[i,5]<mid[5] else data[i,5]-mid[5]
		label[i,0] = 1
	elif((i>=32000 and i<34000) or (i>=116000 and i<117000)):
		data[i,6] = data[i,6]+mid[6] if data[i,6]<mid[6] else data[i,6]-mid[6]
		label[i,0] = 1
	elif((i>=36000 and i<38000) or (i>=118000 and i<119000)):
		data[i,7] = data[i,7]+mid[7] if data[i,7]<mid[7] else data[i,7]-mid[7]
		label[i,0] = 1
	elif((i>=42000 and i<44000) or (i>=122000 and i<123000)):
		data[i,8] = data[i,8]+mid[8] if data[i,8]<mid[8] else data[i,8]-mid[8]
		label[i,0] = 1
	elif((i>=46000 and i<48000) or (i>=124000 and i<125000)):
		data[i,9] = data[i,9]+mid[9] if data[i,9]<mid[9] else data[i,9]-mid[9]
		label[i,0] = 1
	elif((i>=52000 and i<54000) or (i>=126000 and i<127000)):
		data[i,10] = data[i,10]+mid[10] if data[i,10]<mid[10] else data[i,10]-mid[10]
		label[i,0] = 1
	elif((i>=56000 and i<58000) or (i>=128000 and i<129000)):
		data[i,11] = data[i,11]+mid[11] if data[i,11]<mid[11] else data[i,11]-mid[11]
		label[i,0] = 1
	elif((i>=62000 and i<64000) or (i>=132000 and i<133000)):
		data[i,12] = data[i,12]+mid[12] if data[i,12]<mid[12] else data[i,12]-mid[12]
		label[i,0] = 1
	elif((i>=66000 and i<68000) or (i>=134000 and i<135000)):
		data[i,13] = data[i,13]+mid[13] if data[i,13]<mid[13] else data[i,13]-mid[13]
		label[i,0] = 1
	elif((i>=72000 and i<74000) or (i>=136000 and i<137000)):
		data[i,14] = data[i,14]+mid[14] if data[i,14]<mid[14] else data[i,14]-mid[14]
		label[i,0] = 1
	elif((i>=76000 and i<78000) or (i>=138000 and i<139000)):
		data[i,15] = data[i,15]+mid[15] if data[i,15]<mid[15] else data[i,15]-mid[15]
		label[i,0] = 1
	elif((i>=82000 and i<84000) or (i>=142000 and i<143000)):
		data[i,16] = data[i,16]+mid[16] if data[i,16]<mid[16] else data[i,16]-mid[16]
		label[i,0] = 1
	elif((i>=86000 and i<88000) or (i>=144000 and i<145000)):
		data[i,17] = data[i,17]+mid[17] if data[i,17]<mid[17] else data[i,17]-mid[17]
		label[i,0] = 1
	elif((i>=92000 and i<94000) or (i>=146000 and i<147000)):
		data[i,18] = data[i,18]+mid[18] if data[i,18]<mid[18] else data[i,18]-mid[18]
		label[i,0] = 1

np.savetxt('X_train3.txt', data[0:95000,:], delimiter=",", fmt='%.4f')
np.savetxt('y_train3.txt', label[0:95000,0], fmt='%d')
np.savetxt('X_test3.txt', data[95000:data.shape[0],:], delimiter=",", fmt='%.4f')
np.savetxt('y_test3.txt', label[95000:data.shape[0],0], fmt='%d')
