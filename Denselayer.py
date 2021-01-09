import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
import matplotlib.pyplot as plt


wd= 0.00001

def build_model():
  model= Sequential()
  model.add(Flatten(input_shape=(28,28,1)))
  model.add(Dense(500,kernel_initializer='glorot_uniform',use_bias=True, activation='relu',kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Dense(100,kernel_initializer='glorot_uniform',use_bias=True, activation='relu',kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Dense(10,kernel_initializer='glorot_uniform',use_bias=True,activation='softmax',kernel_regularizer=keras.regularizers.l2(wd)))
  opt = optimizers.Adam(lr=.001)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

#load data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(x_test.shape)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mean= np.mean(x_train)
std=np.std(x_train)
x_train=(x_train-mean)/std
x_test=(x_test-mean)/std
x_train=tf.expand_dims(x_train,-1)
x_test=tf.expand_dims(x_test,-1)
print(x_train.shape)
print(x_test.shape)

model=build_model()
print(model.summary())
model.fit(x_train, y_train,batch_size=120,epochs=100, validation_data=(x_test, y_test),shuffle=True)

model.save('Dense.h5')