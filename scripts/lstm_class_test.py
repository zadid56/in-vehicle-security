# lstm model
import numpy as np
import time
import keras_metrics as km
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

trainX = np.loadtxt('X_train.txt', delimiter=" ")
trainy = np.loadtxt('y_train.txt')
testX = np.loadtxt('Xs_test.txt', delimiter=" ")
testy = np.loadtxt('ys_test.txt')

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
testX = testX.reshape(1,1,shape[0])
shape = np.shape(testy)
print(shape)
testy = testy.reshape(1,1)

# fit and evaluate a model

epochs, batch_size, n_neurons, dropout = 50, 128, 50, 0.5
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
_, ba, pr, rec, tp, fp, tn, fn = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
start_eval = time.time()
out = model.predict_classes(testX)
end_eval = time.time() - start_eval
print(end_eval)
#print(ba, pr, rec, end_eval)
#print(hist.history)
