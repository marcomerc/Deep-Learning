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
frauds = np.concatenate(  (mappings[(8,8)],  mappings[(8,10)]) , axis=0 )# 1 means adding vertically list of customers associated with that





