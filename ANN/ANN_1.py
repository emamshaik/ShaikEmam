#Artifical Neural Network
#Part_1 Data preprocessing
#Importing the Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Importing the Dataset
os.chdir("C:\\Users\\Inshema\\Downloads")
dataset=pd.read_csv('Churn_Modelling.csv')
X= dataset.iloc[:, 3:13].values
y=dataset.iloc[:,13].values
#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
#Spliting the dataset into the Training_set and Test_set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# part_2 Now let's make ANN
#Importig the keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Intialising the ANN
classifier= Sequential()
#Adding the Input layer and First Hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#Adding Second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#Adding Output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
