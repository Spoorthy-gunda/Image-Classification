# Image-Classification



## Introduction: 

In this project, I have trained various neural networks for classifying images. The Datasets that were used are MNIST, FMNIST, CIFAR 10, and SVHN. Studied Various regularization techniques their effect while training. 

### Data Normalization:
When the range of the data varies training might get harder while to update the weights. It is best to scale all the inputs to lie in in the range of 0 to 1. There are different ways to normalize data few of them are as follows:
1. Dividing all the data by the maximum possible value, this scales the data in the rnage of 0 to 1.
2. Subtracting minimun value and diving the diffrence of maximum and minimum from data(d-min/(max-min)) is another way to normalize the data that lies in the range of 0 to 1.
3. Most commonly used data normalization is to calculate the mean and standard deviation of the data. Subtract the mean and divide the value by the standard deviation ((d- mean)/std) to make the data have a normal distribution with mean 0 and standard deviation 1.

# Different ways to control overfitting
Overfiting occurs when a network fits the training set with a very high accuracy but cannot be generalized on test or validation sets. One of the reason for a network to overfit is that the network is so big and hence the number of parameters to be learned are a lot more than data available. But smaller networks are not very good at learning most of the datasets. Following regularization techniques can be used to overcome the problem of overfitting

## Regularization:
When a network is overfitting the weights have a high magnitude in order to fit all the training data perfectly but it cannot be generalized. Regularization penalizes the weights and hence prevents the weights from blowing up.
## Data Augmentation:
Large and deep neural networks are useful to learn most of the features but if the data available is less, the network will overfit. This can be overcome by generating data in real-time. New data is generated from the existing by applying various operations such as rotation, cropping,  flipping, and scaling.
## Dropout
During training, some of the nodes are ignored. This results in the training of different networks on the same dataset without increasing the number of parameters to be trained.
 
  
![without_reg](https://user-images.githubusercontent.com/77033276/104082311-08fe1b00-51ea-11eb-9909-8a252ea1e9ef.PNG)
	
Training and testing loss over epochs when no regularization is applied.

![reg](https://user-images.githubusercontent.com/77033276/104082250-82493e00-51e9-11eb-889f-81d6ed95d4d1.PNG)

Training and testing loss over epochs when L2 regularization is used while training.

![dropout_l2reg](https://user-images.githubusercontent.com/77033276/104082255-8aa17900-51e9-11eb-99e4-f3fcef8f5256.PNG)

Training and testing loss over epochs when L2 regularization and Dropout are used while training.

![dropout_reg_bn_da](https://user-images.githubusercontent.com/77033276/104082262-8ffec380-51e9-11eb-9784-e3baffb2a457.PNG)

Training and testing loss over epochs when L2 Regularization, Dropout and Data Augmentation are used while training.



## Datasets:
### MNIST:
This dataset contains Handwritten digits from (0 to 9), 60,000 images are available for training and 10,000 for testing. Images are black white and contain 28x28 pixels which has a value of its intensity.
### FMNIST:
This dataset contains images of 10 different kinds of clothing. These images also are black white and have a single channel of 28x28 pixels.
### SVHN:
This is a real-world dataset of house numbers from google street view images. There are 10 classes for each digit. There is a lot of noise in these images and cropped to fit in a single digit of 32 x32 pixels. Unlike MNISt dataset this has 3 channels for each image representing the intensity of each primary colour(R, G, B).
### CIFAR 10:
This dataset consists of coloured images of 10 different classes. Each image has 3 channels (R,G,B) and 32x32 pixels.

## Network Architectures:
### LENET
LENET is one of the first CNN's that were introduced. It is a simple neural network with 2 convolutional layers, downsampling and follwed by 1 densely connected layers and output layer. 
http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf
### VGG
VGG is a very deep neural network that was implemented for ImageNet. It is deep neural network with kernel size as 3x3 as opposed to the larger kernel sizes of 5x5, 7x7. The authors of the papers showed that large kernel sizes are not necessary for training. Using 3x3 kernel size also reduces the number of parameters to be trained by a good factor. https://arxiv.org/abs/1409.1556
### NiN
In NiN more non-linearity is introduced using mpl convolutions layer(convolution layer with 1x1 kernel) and Global Average pooling instead of a dense layer. The authors of the paper showed that the non-linaerity and global averaging prevents overfitting globally.
https://arxiv.org/abs/1312.4400
 




