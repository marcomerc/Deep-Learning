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
classifier.add(Conv2D(32,3,3, input_shape= (64,64,3), activation= "relu"))  # first number is the number of feature maps the other two is the dimentions 3 by 3
# then input_shape(first how mnay channels color has 3 and back and white has 2 chanels,256 by 256 pixels ) backwords because of tensor 

#pooling reducing the size of the feature map on each of the feauter maps 
classifier.add(MaxPooling2D(pool_size= (2,2) )) #the size of the feature map by two

#this puts all the feature maps into an array and then get it ready to put it in the neural network.
classifier.add(Flatten())

#making a regular neural net
classifier.add(Dense(128, activation = 'relu' ))#first is the output layer size
classifier.add(Dense(1, activation = 'sigmoid'))#first is the output layer size

#compiling 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# re scaling them 
from keras.preprocessing.image import ImageDataGenerator 


#so that we generate more imamges from what we have and trian better we rescalel them and do zooms and stuff 
train_datagen = ImageDataGenerator( 
        rescale=1./255, # reducint value of each pizel 
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64, 64), # this is what the neural net will be expecting
                                              batch_size=32,
                                              class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
                        train_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=800)




