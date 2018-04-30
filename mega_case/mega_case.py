# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the data set
dataset = pd.read_csv('Credit_Card_Applications.csv')
x= dataset.iloc[:,:-1] # the -1 is for the last value 
y= dataset.iloc[:,-1] # the -1 is for the last value 




#fesature scaling  Data preprocessing/
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(x) # NORMALIZATION

# we are going to use a libary.
from minisom import MiniSom
som  = MiniSom(x = 10, y =10, input_len = 15, sigma =1.0, learning_rate = 0.5) 
som.random_weights_init(X)
som.train_random(X, num_iteration = 100)


# we have to visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) # all distances for all the neural nets 
colorbar() #add a bar 
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X): # customers is x 
    w = som.winner(x)      #the wining node for a customer
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    
show()

# finding the fraud.
mappings = som.win_map(X)
frauds = np.concatenate(  (mappings[(8,5)],  mappings[(9,10)]) , axis=0 )# 1 means adding vertically list of customers associated with that
frauds.reshape(1,-1)
frauds = sc.inverse_transform(frauds)

# creating a depending varibale
Customers = dataset.iloc[:,1:].values # the -1 is for the last value 
#
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
        
        
        

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Customers = sc.fit_transform(Customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(Customers, is_fraud, batch_size = 1, epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
                

