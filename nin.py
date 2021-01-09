import keras
import numpy as np
import tensorflow as tf
import hdf5storage as io
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal  
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
import matplotlib.pyplot as plt

batch_size    = 128
epochs        = 150
iterations    = 391
num_classes   = 10
dropout       = 0.5
wd  = 0.00001
log_filepath  = './nin'


    

def scheduler(epoch):
    if epoch <= 60:
        return 0.001
    if epoch <= 70:
        return 0.0001
    if epoch <= 200:    
        return 0.00001
    return 0.0004

def build_model():
  model = Sequential()

  model.add(Conv2D(192, (5,5), padding='same',use_bias=True,kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd), input_shape=x_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Conv2D(160, (1, 1), padding='same', use_bias=True, kernel_initializer='glorot_uniform',kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  model.add(Conv2D(96, (1, 1), padding='same', use_bias=True,kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding = 'same'))
  
  model.add(Dropout(dropout))
  
  model.add(Conv2D(192, (5,5), padding='same', use_bias=True,kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Conv2D(192, (1, 1),padding='same',use_bias=True,kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  model.add(Conv2D(192, (1, 1),padding='same', use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding = 'same'))
  
  model.add(Dropout(dropout))
  
  model.add(Conv2D(192, (5,5), padding='same',use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Conv2D(192, (1, 1), padding='same',use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  model.add(Conv2D(10, (1, 1), padding='same', use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(wd)))
  model.add(Activation('relu'))
  
  model.add(GlobalMaxPooling2D())
  model.add(Activation('softmax'))
  
  opt = optimizers.Adam(lr=.001)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  # sgd = optimizers.SGD(lr=.00001, momentum=0.9, nesterov=True)
  # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  return model

if __name__ == '__main__':

    ## loading data and preprocessing for svhn 
    # train_raw= io.loadmat('/content/svhn_train_32x32.mat')
    # test_raw= io.loadmat('/content/svhn_test_32x32.mat')

    # train_images = np.array(train_raw['X'])
    # test_images = np.array(test_raw['X'])

    # train_labels = train_raw['y']
    # test_labels = test_raw['y']

    # for i in range(train_labels.shape[0]):
      # if train_labels[i]==10:
        # train_labels[i]=0
    # for i in range(test_labels.shape[0]):
      # if test_labels[i]==10:
        # test_labels[i]=0

    # train_images = np.moveaxis(train_images, -1, 0)
    # test_images = np.moveaxis(test_images, -1, 0)

    # y_train = keras.utils.to_categorical(train_labels, num_classes)
    # y_test = keras.utils.to_categorical(test_labels, num_classes)
    # x_train = train_images.astype('float32')
    # x_test = test_images.astype('float32')
    # mean = np.mean(x_train,axis=(0,1,2,3))
    # std = np.std(x_train, axis=(0, 1, 2, 3))        
    # for i in range(3):
        # x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        # x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    
    # loading data for cifar 10
    (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    m=np.max(x_train)
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train, axis=(0, 1, 2, 3))        
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    model = build_model()
    

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

 

    datagen = ImageDataGenerator(rotation_range=8,
                             zoom_range=[0.95, 1.05],
                             height_shift_range=0.10,
                             shear_range=0.15)
    datagen.fit(x_train)


    model=build_model()
    print(model.summary())
    history=model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,epochs=epochs, validation_data=(x_test, y_test),shuffle=True,callbacks=cbks)
    model.save('svhn_nin.h5')


    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'])
    plt.show()
