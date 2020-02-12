# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train = np.loadtxt('X_train3.txt', delimiter=",")
y_train = np.loadtxt('y_train3.txt')
X_test = np.loadtxt('X_test3.txt', delimiter=",")
y_test = np.loadtxt('y_test3.txt')

#xmin = np.array([0,0,0,0,0,0,0,-48,0,-15.0234742,0,0,0,0,0,0,-3276.8,0,0,0])
#xmax = np.array([3,99.6094,16383.75,99.6094,99.6094,254,3,143.25,3,104.6948357,99.603,25.8984375,99.609375,99.609375,99.609375,99.609375,3276.8,1016,15,15])
X_test = (X_test - X_train.min(0)) / X_train.ptp(0)
X_train = (X_train - X_train.min(0)) / X_train.ptp(0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Classification to the Training set
class_weights = {0: 1,1: 5}
aprf = np.zeros((30,4))
for i in range(1):
	classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', class_weight=class_weights)
	classifier.fit(X_train, y_train)
	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	# Making the Confusion Matrix
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)
	aprf[i,0] = (tp+tn)/(tp+fp+tn+fn)
	aprf[i,1] = tp/(tp+fp)
	aprf[i,2] = tp/(tp+fn)
	aprf[i,3] = 2*aprf[i,1]*aprf[i,2]/(aprf[i,1]+aprf[i,2])
	
np.savetxt('/home/mdzadik/CAN_data/pickles/aprf_rf.csv',aprf,delimiter=",")
