# Udacity CVND Image Captioning Project
Udacity Computer Vision Nanodegree Image Captioning Project

## Introduction
The repository contains a neural network, which can automatically generate captions from images. 

## Network Architecture
The solution architecture consists of:
1. CNN encoder, which encodes the images into the embedded feature vectors:
![image](https://github.com/Lexie88rus/Udacity-CVND-Image-Captioning/raw/master/assets/encoder.png)
2. Decoder, which is a sequential neural network consisting of LSTM units, which translates the feature vector into a sequence of tokens:
![image](https://github.com/Lexie88rus/Udacity-CVND-Image-Captioning/raw/master/assets/decoder.png)

## Results
These are some of the outputs give by the network using the [COCO dataset](http://cocodataset.org/):
![example1](https://github.com/Lexie88rus/Udacity-CVND-Image-Captioning/raw/master/assets/example1.png)
![example2](https://github.com/Lexie88rus/Udacity-CVND-Image-Captioning/raw/master/assets/example2.png)
