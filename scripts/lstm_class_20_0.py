# lstm model
import numpy as np
import pickle
import keras_metrics as km
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM

trainX = np.loadtxt('X_train.txt', delimiter=" ")
trainy = np.loadtxt('y_train.txt')
testX = np.loadtxt('X_test.txt', delimiter=" ")
testy = np.loadtxt('y_test.txt')

xmin = np.array([0,0,0,0,0,0,0,-48,0,-15.0234742,0,0,0,0,0,0,-3276.8,0,0,0])
xmax = np.array([3,99.6094,16383.75,99.6094,99.6094,254,3,143.25,3,104.6948357,99.603,25.8984375,99.609375,99.609375,99.609375,99.609375,3276.8,1016,15,15])
xptp = xmax - xmin
trainX = (trainX - xmin) / xptp
testX = (testX - xmin) / xptp

shape = np.shape(trainX)
trainX = trainX.reshape(shape[0],1,shape[1])
shape = np.shape(trainy)
trainy = trainy.reshape(shape[0],1)
shape = np.shape(testX)
testX = testX.reshape(shape[0],1,shape[1])
shape = np.shape(testy)
testy = testy.reshape(shape[0],1)

# fit and evaluate a model
batch_size_arr = [32,128,512,1024]
epochs_arr = [50,200,500]
ba = np.zeros((1,4))
pr = np.zeros((1,4))
rec = np.zeros((1,4))
tp = np.zeros((1,4))
fp = np.zeros((1,4))
tn = np.zeros((1,4))
fn = np.zeros((1,4))
for i in range(1):
	for j in range(4):
		epochs, batch_size, n_neurons, dropout = epochs_arr[i], batch_size_arr[j], 20, 0
		n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
		model = Sequential()
		model.add(LSTM(n_neurons, input_shape=(n_timesteps,n_features)))
		model.add(Dropout(dropout))
		model.add(Dense(n_neurons, activation='relu'))
		model.add(Dense(n_outputs, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy', km.binary_precision(), km.binary_recall(), km.binary_true_positive(), km.binary_false_positive(), km.binary_true_negative(), km.binary_false_negative()])
		# fit network
		model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)
		# evaluate model
		_, ba[i,j], pr[i,j], rec[i,j], tp[i,j], fp[i,j], tn[i,j], fn[i,j] = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
		#print(ba, pr, rec, tp, fp, tn, fn)
		#print(hist.history)

np.savetxt('/home/mdzadik/CAN_data/pickles/ba_'+str(n_neurons)+'_'+str(dropout)+'.csv',ba,delimiter=",")
np.savetxt('/home/mdzadik/CAN_data/pickles/pr_'+str(n_neurons)+'_'+str(dropout)+'.csv',pr,delimiter=",")
np.savetxt('/home/mdzadik/CAN_data/pickles/rec_'+str(n_neurons)+'_'+str(dropout)+'.csv',rec,delimiter=",")
np.savetxt('/home/mdzadik/CAN_data/pickles/tp_'+str(n_neurons)+'_'+str(dropout)+'.csv',tp,delimiter=",")
np.savetxt('/home/mdzadik/CAN_data/pickles/fp_'+str(n_neurons)+'_'+str(dropout)+'.csv',fp,delimiter=",")
np.savetxt('/home/mdzadik/CAN_data/pickles/tn_'+str(n_neurons)+'_'+str(dropout)+'.csv',tn,delimiter=",")
np.savetxt('/home/mdzadik/CAN_data/pickles/fn_'+str(n_neurons)+'_'+str(dropout)+'.csv',fn,delimiter=",")


# n_neurons = 50
# dropout = 0.5
# pkl_file = open('/home/mdzadik/CAN_data/pickles/ba_'+str(50)+'_'+str(0)+'.pkl', 'rb')
# ba = pickle.load(pkl_file)
# pkl_file.close()

# pkl_file = open('/home/mdzadik/CAN_data/pickles/pr_'+str(50)+'_'+str(0)+'.pkl', 'rb')
# pr = pickle.load(pkl_file)
# pkl_file.close()

# pkl_file = open('/home/mdzadik/CAN_data/pickles/rec_'+str(50)+'_'+str(0)+'.pkl', 'rb')
# rec = pickle.load(pkl_file)
# pkl_file.close()

# pkl_file = open('/home/mdzadik/CAN_data/pickles/tp_'+str(50)+'_'+str(0)+'.pkl', 'rb')
# tp = pickle.load(pkl_file)
# pkl_file.close()

# pkl_file = open('/home/mdzadik/CAN_data/pickles/fp_'+str(50)+'_'+str(0)+'.pkl', 'rb')
# fp = pickle.load(pkl_file)
# pkl_file.close()

# pkl_file = open('/home/mdzadik/CAN_data/pickles/tn_'+str(50)+'_'+str(0)+'.pkl', 'rb')
# tn = pickle.load(pkl_file)
# pkl_file.close()

# pkl_file = open('/home/mdzadik/CAN_data/pickles/fn_'+str(50)+'_'+str(0)+'.pkl', 'rb')
# fn = pickle.load(pkl_file)
# pkl_file.close()


# repeat experiment
# scores = list()
# for r in range(repeats):
	# score = evaluate_model(trainX, trainy, testX, testy)
	# score = score * 100.0
	# print('>#%d: %.3f' % (r+1, score))
	# scores.append(score)
	
# print(scores)
# m, s = mean(scores), std(scores)
# print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))






# import numpy as np
# import pandas as pd
# #df = pd.read_csv('attack_free_dataset.txt', sep='\s+', header=None, skiprows=1)
# df = pd.read_csv('/home/mdzadik//CAN_data/attack_data/test.csv', header=None)

# data = np.zeros((1,20))
# timestamp = np.zeros((1))
# count = 0

# for i in range(0,df.shape[0]):
	# code = int(df.iloc[i,0],16)
	# if(code==790):
		# data = np.append(data,data[count,:].reshape(1,20),axis=0)
		# temp = df.iloc[i,:].reshape(1,11)
		# timestamp = np.append(timestamp,temp[0,10])
		# count = count+1
		# print(count)
		# data[count,0] = int(bin(int(temp[0,2],16))[2:].zfill(8)[4:6],2)
		# data[count,1] = int(temp[0,3],16)*0.390625
		# data[count,2] = (int(temp[0,4],16)*256+int(temp[0,5],16))*0.25
		# data[count,3] = int(temp[0,6],16)*0.390625
		# data[count,4] = int(temp[0,7],16)*0.390625
		# data[count,5] = int(temp[0,8],16)
		
	# if(code==809):
		# data = np.append(data,data[count,:].reshape(1,20),axis=0)
		# temp = df.iloc[i,:].reshape(1,11)
		# timestamp = np.append(timestamp,temp[0,10])
		# count = count+1
		# print(count)
		# data[count,6] = int(bin(int(temp[0,2],16))[2:].zfill(8)[6:8],2)
		# data[count,7] = int(temp[0,3],16)*0.75 - 48
		# data[count,8] = int(bin(int(temp[0,6],16))[2:].zfill(8)[0:2],2)
		# data[count,9] = int(temp[0,7],16)*0.4694836 - 15.0234742
		# data[count,10] = int(temp[0,8],16)*0.3906
		
	# if(code==1349):
		# data = np.append(data,data[count,:].reshape(1,20),axis=0)
		# temp = df.iloc[i,:].reshape(1,11)
		# timestamp = np.append(timestamp,temp[0,10])
		# count = count+1
		# print(count)
		# data[count,11] = int(temp[0,5],16)*0.1015625
		
	# if(code==608):
		# data = np.append(data,data[count,:].reshape(1,20),axis=0)
		# temp = df.iloc[i,:].reshape(1,11)
		# timestamp = np.append(timestamp,temp[0,10])
		# count = count+1
		# print(count)
		# data[count,12] = int(temp[0,2],16)*0.390625
		# data[count,13] = int(temp[0,3],16)*0.390625
		# data[count,14] = int(temp[0,4],16)*0.390625
		# data[count,15] = int(temp[0,7],16)*0.390625
		
	# if(code==688):
		# data = np.append(data,data[count,:].reshape(1,20),axis=0)
		# temp = df.iloc[i,:].reshape(1,11)
		# timestamp = np.append(timestamp,temp[0,10])
		# count = count+1
		# print(count)
		# if(int(bin(int(temp[0,2],16))[2:].zfill(8)[0],2)==0):
			# data[count,16] = (int(bin(int(temp[0,2],16))[2:].zfill(8)[1:8],2)*256+int(temp[0,3],16))*0.1
		# else:
			# data[count,16] = (0-(int(bin(int(temp[0,2],16))[2:].zfill(8)[1:8],2)*256+int(temp[0,3],16)))*0.1
		# data[count,17] = int(temp[0,4],16)*4
		# data[count,18] = int(bin(int(temp[0,6],16))[2:].zfill(8)[0:4],2)
		# data[count,19] = int(bin(int(temp[0,6],16))[2:].zfill(8)[4:8],2)

# np.savetxt('X_test.txt', data, delimiter=" ", fmt='%1.4f')
# np.savetxt('timestamp.txt', timestamp, delimiter=" ", fmt='%1.5f')

# gt = np.zeros((1))

# # for i in range(0,count):
	# # if(((timestamp[i]//10)%2==1 and timestamp[i]<160) or ((timestamp[i]//10)%2==0 and timestamp[i]>170 and timestamp[i]<250)):
		# # gt = np.append(gt,1)
	# # else:
		# # gt = np.append(gt,0)

# for i in range(0,count):
	# if((timestamp[i]//5)%2==1 and timestamp[i]<470):
		# gt = np.append(gt,1)
	# else:
		# gt = np.append(gt,0)		

# np.savetxt('y_test.txt', gt, delimiter=" ", fmt='%-4d')
