# EE-399-HW4
``Author: Ben Li``
``Date: 5/7/2023``
``Course: SP 2023 EE399``

![nagesh-pca-1](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/Comp-1.gif)

## Abstract
This homework is devided into two parts. In part1, we are provided with a a series of data
```python
X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
and we are asked to trains a simple neural network to predict a series of numbers. We will seperate these data into testing data and training data. The performance of this model is then evaluated.

Next, we are ask to perform PCA on a dataset of handwritten digits from MNIST database. Here, we will train the neural network to classify the digits using reduced dimensionality data. We will experiment with the neural network architecture and hyperparameters to try to improve its performance. 

## Introduction and Overview
This assignment involves working with two dataset and apply neural network to them.

The first data set has 31 data points. We need to fit the data to a three layer feed forward neural network. We then split the data into training and testing data and evaluate the performance using least square error.
$$E = \sqrt{(1/n)*\sum_{j=1}^n (f(x_j)-y_j)^2}$$

Then we use the MNIST dataset, which is a series of handwritten digit. We need to compute the first 20 PCA modes of images and build a feed-forward neural network to classify the digits. We also need to compute the results of the neural network with other classifiers such as LSTM, SVM, and decision trees. 

## Theoretical Background
The first important 
## Algorithm Implementation and Development 

## Computational Results

## Summary and Conclusions
