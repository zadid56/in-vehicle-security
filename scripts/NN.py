# lstm model
import numpy as np
import pickle
import keras_metrics as km
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import adamax

trainX = np.loadtxt('X_train3.txt', delimiter=",")
trainy = np.loadtxt('y_train3.txt')
testX = np.loadtxt('X_test3.txt', delimiter=",")
testy = np.loadtxt('y_test3.txt')

#xmin = np.array([0,0,0,0,0,0,0,-48,0,-15.0234742,0,0,0,0,0,0,-3276.8,0,0,0])
#xmax = np.array([3,99.6094,16383.75,99.6094,99.6094,254,3,143.25,3,104.6948357,99.603,25.8984375,99.609375,99.609375,99.609375,99.609375,3276.8,1016,15,15])
testX = (testX - trainX.min(0)) / trainX.ptp(0)
trainX = (trainX - trainX.min(0)) / trainX.ptp(0)

#shape = np.shape(trainX)
#trainX = trainX.reshape(shape[0],1,shape[1])
#shape = np.shape(trainy)
#trainy = trainy.reshape(shape[0],1)
#shape = np.shape(testX)
#testX = testX.reshape(shape[0],1,shape[1])
#shape = np.shape(testy)
#testy = testy.reshape(shape[0],1)

# fit and evaluate a model
batch_size_arr = [1024,32,128,512,1024]
epochs_arr = [300,200,500]
aprf = np.zeros((30,4))
for i in range(1):
	n_timesteps, n_features, n_outputs = 1, trainX.shape[1], 1
	epochs, batch_size, n_neurons, dropout = 300, 512, n_features, 0.5
	n_timesteps, n_features, n_outputs = 1, trainX.shape[1], 1
	model = Sequential()
	model.add(Dense(n_neurons, input_dim=n_features, activation='relu'))
	#model.add(Dropout(dropout))
	model.add(Dense(int(n_neurons/2), activation='relu'))
	model.add(Dropout(dropout))
	model.add(Dense(5, activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	opt = adamax(lr=0.0002)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.save("model_NN.h5")
	# fit network
	checkpoint = ModelCheckpoint('weights_NN.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
	class_weights = {0: 1,1: 2}
	history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(testX,testy), callbacks=[checkpoint], class_weight=class_weights)
	# evaluate model
	model2 = load_model('model_NN.h5')
	model2.load_weights('weights_NN.h5')
	predy = model2.predict(testX)
	predy = np.where(predy > 0.5, 1, 0)
	tn, fp, fn, tp = confusion_matrix(testy, predy).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	aprf[i,0] = (tp+tn)/(tp+fp+tn+fn)
	aprf[i,1] = tp/(tp+fp)
	aprf[i,2] = tp/(tp+fn)
	aprf[i,3] = 2*aprf[i,1]*aprf[i,2]/(aprf[i,1]+aprf[i,2])
	#print("tn, fp, fn, tp: ", tn[i,j], fp[i,j], fn[i,j], tp[i,j])
	# summarize history for loss
	#plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	#plt.title('model loss')
	#plt.ylabel('loss')
	#plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	#_, ba[i,j], pr[i,j], rec[i,j], tp[i,j], fp[i,j], tn[i,j], fn[i,j] = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	#print(ba, pr, rec, tp, fp, tn, fn)
	#print(hist.history)

#print("acc(mean),acc(std),pr(mean),pr(std),rec(mean),rec(std): ", np.mean(acc),np.std(acc),np.mean(pr),np.std(pr),np.mean(rec),np.std(rec))
#np.savetxt('/home/mdzadik/CAN_data/pickles/aprf_nn.csv',aprf,delimiter=",")