# Program: 2 Layer Neural Network [1 input layer, 1 hidden layer and 1 output layer]
# Dataset: MNIST [Handwritten Digits]
# Purpose: To identify the digit from MNIST dataset using NN built in PyTorch

import torch
import torchvision
import numpy as numpy
import matplotlib.pyplot as matplotlib
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Check if there is CUDA on the machine or no
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

# Before downloading dataset setting transformation
# Mean = 0.5 and Standard Deviation = 0.5
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
# print (transform)
# transforms.ToTensor() - This converts the images into numbers and separates into 3 color channels[RGB]
#                         then it converts pixel of each image to brightness of their color between 0 and 255.
# ransforms.Normalize() - It normalizes by taking the mean and standard deviation value. This helps data to be
#                         within range and reduce skewness, this helps processing faster


# MNIST Dataset
# create a dataset and load 
print ("Loading Data for Training")
training = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transform)

print ("Loading Data for Testing")
testset = torchvision.datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

print ("Loading Training Data from training dataset")
trainloader = torch.utils.data.DataLoader(training, batch_size=64, shuffle=True)

print ("Loading Test Data for testset")
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# 64 images in each batch having dimension as 28x28 pixels


# Lets randomly display some 40 images from the dataset
data_iterator = iter(trainloader)
images, labels = data_iterator.next()

# create a new figure
display_figure = matplotlib.figure(num='MNIST Dataset')
# delibrately keeping as 41 as range starts from 1
number_of_images = 41 

for index in range(1, number_of_images):
    matplotlib.subplot(4, 10, index)
    matplotlib.axis('off')
    matplotlib.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    display_figure.suptitle('Randomly display some 40 images from the MNIST Dataset', fontsize=13)
# display the figure
matplotlib.show()