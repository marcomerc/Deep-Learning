
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()                          # we encode the country into a 0 1 2 to make it easier for the net one object to encode       
labelencoder_X_2 = LabelEncoder()                                         # we encode the country into a 0 1 2 to make it easier for the net one object to encode
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # the first endtry is all rows and 1 index or second colomn after this line it's encoded
X[:, 2]= labelencoder_X_2.fit_transform(X[:, 2]) # the first endtry is all rows and 2 index or second colomn after this line it's encoded
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray() # devide the contry colomn into 3 to make it independent 
X = X[:,1:] #then we get rid of one of the new colomns created


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the neural network

#importing libraries 
import keras 
from keras.models import Sequential 
from keras.layers import Dense



#initilizing the neural net
classifier = Sequential()   #geting an object to initizeing the net
 

#first hidden layer
# adding th layers sicne 11 independent variables then we get 11 first notes in the first layers.
# we have to choose the activatio neural network: we choose the retifier for the hidden and the sigmoid for the output since is binary
classifier.add(Dense(  activation='relu',input_dim=11, units= 6, kernel_initializer='uniform' )) # adding layers in hidden layer the average in the input and the output otherwise 11+1/2 we have to do cross validation
#second hidden layer #get rid of units or the first entries becuase it will read from the previosu
#uints is the output layer
classifier.add(Dense(  activation='relu', units= 6, kernel_initializer='uniform' )) # adding layers in hidden layer the average in the input and the output otherwise 11+1/2 we have to do cross validation
#final layer 
classifier.add(Dense(  activation='sigmoid', units= 1, kernel_initializer='uniform' ))
#compile the network 
# adam is stochastic gradient decent, then loss function
# loss function vary if it's none binary
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) # adam is stochastic gradient decent, then loss function




# Fitting classifier to the Training set
# training by bash 
classifier.fit(X_train, y_train, batch_size=10, epochs=100)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 