# Image-Classification



## Introduction: 

In this project, I have trained various neural networks for classifying images. The Datasets that were used are MNIST, FMNIST, CIFAR 10, and SVHN. Studied Various regularization techniques their effect while training. 

Data Normalization: When the range of the data varies training might get harder while to update the weights. It is best to scale all the inputs to lie in in the range of 0 to 1. There are different ways to normalize data few of them are as follows:
Dividing all the data by the maximum possible value, this scale the data from 0 to 1.
d-min/(max-min) is another way to normalize the data that lies in the range of 0 to 1.
Most commonly used data normalization is to calculate the mean and standard deviation of the data. Subtract the mean and divide the value by the standard deviation to make the data have a normal distribution with mean 0 and standard deviation 1.




Usually, networks tend to overfit. One of the reason for a network to overfit is that the network is so big and hence the number of parameters to be learned are a lot more than data available. But smaller networks are not very good at learning most of the datasets. Following regularization techniques can be used to overcome the problem of overfitting

# Regularization:
When a network is overfitting the weights have a high magnitude in order to fit all the training data perfectly but it cannot be generalized. Regularization penalizes the weights and hence prevents the weights from blowing up.
# Data Augmentation:
Large and deep neural networks are useful to learn most of the features but if the data available is less, the network will overfit. This can be overcome by generating data in real-time. New data is generated from the existing by applying various operations such as rotation, cropping,  flipping, and scaling.
# Dropout
During training, some of the nodes are ignored. This results in the training of different networks on the same dataset without increasing the number of parameters to be trained.
 
  
	
Training and testing loss over epochs when no regularization is applied.


Training and testing loss over epochs when L2 regularization is used while training.


Training and testing loss over epochs when L2 regularization and Dropout are used while training.



Training and testing loss over epochs when L2 Regularization, Dropout and Data Augmentation are used while training.


Network Architectures:
Lenet
Vgg
nin

Datasets:
MNIST:
This dataset contains Handwritten digits from (0 to 9), 60,000 images are available for training and 10,000 for testing. Images are black white and contain 28x28 pixels which has a value of its intensity.
FMNIST:
This dataset contains images of 10 different kinds of clothing. These images also are black white and have a single channel of 28x28 pixels.
SVHN:
This is a real-world dataset of house numbers from google street view images. There are 10 classes for each digit. There is a lot of noise in these images and cropped to fit in a single digit of 32 x32 pixels. Unlike MNISt dataset this has 3 channels for each image representing the intensity of each primary colour(R, G, B).
CIFAR 10:
This dataset consists of coloured images of 10 different classes. Each image has 3 channels(R,G,B) and 32x32 pixels.
 




