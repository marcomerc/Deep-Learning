# Marco Mercado 


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D # package for the convolutional layers. there is a 3D dimention look at documentation
from keras.layers import MaxPooling2D #this is for pooling the features for the layers
from keras.layers import Flatten # converting pool feature maps into the feature vector
from keras.layers import Dense # it's for the neural network.

#initializing the network
classifier  = Sequential()

#feature detecting it gets a metrix and gives you a feature map and detects a feature.
classifier.add(Convolution2D())











