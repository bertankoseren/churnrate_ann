# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author: bkoseren
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Artificial_Neural_Networks\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN
clasifier = Sequential()
#input layer
clasifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))
#hidden layer
clasifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
#output layer
clasifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))
#compiling ann
clasifier.compile(optimizer = 'SGD',loss = 'binary_crossentropy',metrics = ['accuracy'])
#fitting ann to training set
clasifier.fit(X_train, y_train,batch_size=25,nb_epoch = 500)


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = clasifier.predict(X_test)
y_pred_2 = (y_pred>0.5)

new_prediction = clasifier.predict(sc.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction2 = (new_prediction > 0.5)

#print the churn rate & boolean value with the customers
a = 0
excel = open('Artificial_Neural_Networks\Churn_Modelling.csv','r')
for line in excel:
    if ("RowNumber" in line):
        pass
    else:   
        prediction_rate = (clasifier.predict(sc.transform(np.array([[X[a][0],X[a][1],X[a][2],X[a][3],X[a][4],X[a][5],X[a][6],X[a][7],X[a][8],X[a][9],X[a][10]]]))))
        prediction = ((prediction_rate)>0.5)
        #prediction_rate = ((prediction_rate*100))
        prediction_rate = float("%0.2f" % (prediction_rate*10))
        print (str(a) + "," + line + "," + (str(prediction_rate).replace("[","") + "%," + str(prediction).replace("[","")).replace("]",""))
        a = a +1
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_2)
