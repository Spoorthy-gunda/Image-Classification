import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, Dropout
from keras.callbacks import LearningRateScheduler, TensorBoard
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


wd=0.0001
batch_size=120
iterations=417
epochs=100

def build_model():
  model= Sequential()
  model.add(Conv2D(32,(5,5),padding='same',use_bias=True,kernel_initializer='glorot_uniform',activation='relu',input_shape=(32,32,3),kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Dropout(0.4))
  model.add(MaxPooling2D((2,2),strides=(2,2)))
  model.add(Conv2D(64,(5,5),padding='same',use_bias=True,kernel_initializer='glorot_uniform',activation='relu',kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Dropout(0.4))
  model.add(MaxPooling2D((2,2),strides=(2,2)))
  model.add(Flatten())
  model.add(Dense(1000,kernel_initializer='glorot_uniform',use_bias=True, activation='relu',kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Dropout(0.4))  
  model.add(Dense(10,kernel_initializer='glorot_uniform',use_bias=True,activation='softmax',kernel_regularizer=keras.regularizers.l2(wd)))
  opt = optimizers.Adam(lr=.001)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

#load data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
m=np.max(x_train)
# x_train=(x_train)/m
# x_test=(x_test)/m
 mean = np.mean(x_train,axis=(0,1,2,3))
 std = np.std(x_train, axis=(0, 1, 2, 3))
for i in range(3):
  x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
  x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]



def scheduler(epoch):
    if epoch <= 10:
        return 0.001
    if epoch <= 60:
        return 0.0001
    if epoch <= 200:    
        return 0.00001
    return 0.0004
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr]

datagen = ImageDataGenerator(            
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images horizantally
            vertical_flip=False, cval=0)  # randomly flip images vertically
       
datagen.fit(x_train)


model=build_model()
print(model.summary())
history=model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,epochs=epochs, validation_data=(x_test, y_test),shuffle=True,callbacks=cbks)

model.save('cifar10_lenet.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'val'])
plt.show()
