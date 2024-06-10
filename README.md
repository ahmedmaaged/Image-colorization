# Image Colorization from Greyscale Using Autoencoders

This repository contains the code for an image colorization project using autoencoders in deep learning. The project focuses on converting greyscale images to color images using the CIFAR-10 dataset and is implemented using Keras in Python.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)

## Introduction

Image colorization is a fascinating application of deep learning where the goal is to add color to black and white images. This project utilizes autoencoders to achieve image colorization. Autoencoders are neural networks that aim to learn a compressed representation of input data and then reconstruct the data from this representation.

In this project, the CIFAR-10 dataset is used, which consists of 60,000 32x32 color images in 10 different classes.

## Dataset

The CIFAR-10 dataset is a widely used dataset for machine learning and computer vision tasks. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

For this project, the images are first converted to greyscale and then used as input to the autoencoder. The autoencoder then learns to predict the color version of the greyscale images.

## Model Architecture

The model architecture for this project consists of a convolutional autoencoder. The encoder part of the network compresses the input greyscale image into a lower-dimensional representation, and the decoder part reconstructs the color image from this representation. The architecture can be summarized as follows:

### Encoder
The encoder compresses the input image into a compact latent representation. It consists of several convolutional layers that progressively reduce the spatial dimensions of the input while increasing the number of feature maps.

- **Conv2D Layer**: Applies a convolution operation to the input data.
- **BatchNormalization Layer**: Normalizes the activations of the previous layer.
- **Activation Layer**: Applies a non-linear activation function (e.g., ReLU).
- **MaxPooling2D Layer**: Downsamples the input by taking the maximum value over an input window.

### Decoder
The decoder reconstructs the color image from the compact latent representation. It consists of several convolutional layers that progressively increase the spatial dimensions of the input while decreasing the number of feature maps.

- **Conv2DTranspose Layer**: Applies a transposed convolution operation to the input data.
- **BatchNormalization Layer**: Normalizes the activations of the previous layer.
- **Activation Layer**: Applies a non-linear activation function (e.g., ReLU).
- **UpSampling2D Layer**: Upsamples the input by repeating the rows and columns.

### Model Architecture Diagram
![Autoencoders Architecture](assets/autoencoder_architecture.png)
