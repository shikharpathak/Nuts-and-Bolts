# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:23:20 2017

@author: Shikhar
"""

import Matrix_CV_ML3D as DImage
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')

x = DImage.Matrix_CV_ML3D("G:/Deep Learning/BoltsNuts/train",65,50)
x.build_ML_matrix()


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


y = np_utils.to_categorical(x.labels)
x = x.global_matrix
x = x.astype('float32')/255

model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3,50,65)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
   
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(2, activation='softmax'))
 
# 8. Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(x, y, nb_epoch=80, verbose=2,batch_size=32)
 
# 10. Evaluate model on test data
score = model.evaluate(x, y, verbose=0)
