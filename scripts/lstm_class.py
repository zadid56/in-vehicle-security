# lstm model
import numpy as np
import pickle
import keras_metrics as km
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from sklearn.metrics import confusion_matrix
from keras.optimizers import adamax
import matplotlib.pyplot as plt
from keras.layers import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

trainX = np.loadtxt('X_train3.txt', delimiter=",")
trainy = np.loadtxt('y_train3.txt')
testX = np.loadtxt('X_test3.txt', delimiter=",")
testy = np.loadtxt('y_test3.txt')

#xmin = np.array([0,0,0,0,0,0,-48,0,-15.0234742,0,0,0,0,0,0,-3276.8,0,0,0])
#xmax = np.array([99.6094,16383.75,99.6094,99.6094,254,3,143.25,3,104.6948357,99.603,25.8984375,99.609375,99.609375,99.609375,99.609375,3276.8,1016,15,15])
testX = (testX - trainX.min(0)) / trainX.ptp(0)
trainX = (trainX - trainX.min(0)) / trainX.ptp(0)

shape = np.shape(trainX)
trainX = trainX.reshape(shape[0],1,shape[1])
shape = np.shape(trainy)
trainy = trainy.reshape(shape[0],1)
shape = np.shape(testX)
testX = testX.reshape(shape[0],1,shape[1])
shape = np.shape(testy)
testy = testy.reshape(shape[0],1)

# fit and evaluate a model
batch_size_arr = [16,32,128,512,1024]
epochs_arr = [300,200,500]
aprf = np.zeros((30,4))
for i in range(1):
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	epochs, batch_size, n_neurons, dropout = 300, 512, n_features, 0.5
	model = Sequential()
	model.add(LSTM(n_neurons, activation='relu', input_shape=(n_timesteps,n_features), return_sequences=True))
	model.add(LSTM(int(n_neurons/2), activation='relu', return_sequences=False))
	model.add(Dropout(dropout))
	#model.add(Dense(10, activation='relu'))
	#model.add(Dropout(dropout))
	model.add(Dense(5, activation='relu'))
	#model.add(Dropout(dropout))
	model.add(Dense(n_outputs, activation='sigmoid'))
	opt = adamax(lr=0.0005)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.save("model_lstm.h5")
	# fit network
	#early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
	checkpoint = ModelCheckpoint('weights_lstm.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
	class_weights = {0: 1,1: 2}
	history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(testX,testy), callbacks=[checkpoint], class_weight=class_weights)
	# evaluate model
	model2 = load_model('model_lstm.h5')
	model2.load_weights('weights_lstm.h5')
	predy = model2.predict(testX)
	predy0 = np.where(predy > 0.5, 1, 0)
	predy1 = np.where(predy > 0.45, 1, 0)
	predy2 = np.where(predy > 0.4, 1, 0)
	predy3 = np.where(predy > 0.35, 1, 0)
	predy4 = np.where(predy > 0.3, 1, 0)
	predy5 = np.where(predy > 0.25, 1, 0)
	predy6 = np.where(predy > 0.2, 1, 0)
	predy7 = np.where(predy > 0.15, 1, 0)
	predy8 = np.where(predy > 0.1, 1, 0)
	tn, fp, fn, tp = confusion_matrix(testy, predy0).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy1).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy2).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy3).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy4).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy5).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy6).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy7).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	tn, fp, fn, tp = confusion_matrix(testy, predy8).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	aprf[i,0] = (tp+tn)/(tp+fp+tn+fn)
	aprf[i,1] = tp/(tp+fp)
	aprf[i,2] = tp/(tp+fn)
	aprf[i,3] = 2*aprf[i,1]*aprf[i,2]/(aprf[i,1]+aprf[i,2])
	#print("tn, fp, fn, tp: ", tn[i,j], fp[i,j], fn[i,j], tp[i,j])
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

#print("acc(mean),acc(std),pr(mean),pr(std),rec(mean),rec(std): ", np.mean(acc),np.std(acc),np.mean(pr),np.std(pr),np.mean(rec),np.std(rec))
#np.savetxt('/home/mdzadik/CAN_data/pickles/aprf_lstm.csv',aprf,delimiter=",")
