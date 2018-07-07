#Data preprocessing
#Imorting the Libraries
import numpy as np
import matplotlib.pyplot as plt
#Importing the Dataset
dataset=pd.read_csv('dataset_name.csv')
X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#Spliting the dataset into the Training_set and Test_set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
