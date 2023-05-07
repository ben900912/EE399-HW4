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
The first important concept that is involved in this assignment is **feedforward neural netowrks**. 
[Feed foward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) is a mechanism that the inpute signals gets fed foward into a neural network. It pass through different layers of the network in form of activations and results in the form of some classfiication on the output layer. Below is a animatin illustrating feed forward neural network 
![nagesh-pca-1](https://vitalflux.com/wp-content/uploads/2020/10/feed_forward_neural_network-1.gif)

1. **Layers** The figure shown alove has four different layers. one input layer, two hidden layer, and one output layer. 
2. **Input fed into input layer** there are four input variables which are fed into different nodes in the neural network throug input layer.
3. **Activations in the hidden layer** the sum of input signals combined with weights and bias element are fed into all neurons of the hidden layers.
in each node, all incoming values are added together and fed into an activation function. 
4. **Output in the final layer** In the last step, the activation signals from the hidden layers are combined with weights and fed into the output layer. At each node, all the incoming values are added together in different nodes and then processed with a function to output the probabilities. 

### There are also other important concept that are previously used in other assignments. Since it was introduced before, I will beiefly give a quick explaination on these concepts. 
1. **Principle component analysis**: This is used for dimentionality reduction. It transforms a dataset of potentially correlated variables into a set of linearly uncorrelated variables.
2. **Support vector machines**: This is an algorithm that is used for classification and regression. It finds a hyperplane that maximizes the margin between the classes in data. 
3. **Decision trees**: It works by recursively partitioning the data based on the values of the input features in order to maximize the informatin gain at each node.

## Algorithm Implementation and Development 
There are several steps involved into complete this assignment 

First, we are asked to fit the data to a three layer feed forward neural network. We need to import the necessary libraries and create a network using the Keras library. 

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

X = np.arange(0,31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# Define the model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(X, Y, epochs=1000, verbose=0)
```

> This is a neural network with 10 neurons in the input layer, 5 neurons in the hidden layer, and 1 neuron in the output layer. The model is compiled using the mean squared error loss function. The model is then trained for 100 epochs on the input data. 

Next, we can fit the neural network using the first 20 data points as training data and the remaining 10 data as test dtata. Then we can compute the least square error of these models on the test data using the following code. 
```python
# Split data into training and test sets
X_train, Y_train = X[:20], Y[:20]
X_test, Y_test = X[20:], Y[20:]

# Fit the model to the training data
model.fit(X_train, Y_train, epochs=1000, verbose=0)

# Compute the least square error for the training and test data
train_error = np.mean(np.square(model.predict(X_train).flatten() - Y_train))
test_error = np.mean(np.square(model.predict(X_test).flatten() - Y_test))

print("Training Error: {:.2f}".format(train_error))
print("Test Error: {:.2f}".format(test_error))
```

we can also repeat this process using the first 10 and last 10 data points as training data

```python
X_train, Y_train = np.concatenate((X[:10], X[20:])), np.concatenate((Y[:10], Y[20:]))
X_test, Y_test = X[10:20], Y[10:20]
```

Now to tarain a feedforward neural network on MNIST dataset. 
To compute the first 20 PCA modes of the MNIST images, you cna preprocess the data by substracting the mean of the training set from eacch image. Then, we use SVD to compute the eigenvectors and eigenvalues of the covariance matrix of the preprocessed training data. 

```python
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to 2D and preprocess by subtracting mean
X_train_flat = X_train.reshape(-1, 784) - X_train.reshape(-1, 784).mean(axis=0)

# Compute first 20 PCA modes
pca = PCA(n_components=20)
pca.fit(X_train_flat)
X_train_pca = pca.transform(X_train_flat)
```

We use the same method to a feedforward neural netowkr to classify the digits using Keras API. We also use [scikit-learn](https://scikit-learn.org/) to build a LSTM, SVM, and decision tree classfiiers so that we can compare the result later. 

## Computational Results
From the first task, we can use the result using two different datasets to compare against each other. Specifically, we can compare the models 

![image](https://user-images.githubusercontent.com/121909443/236701478-d0b12cc2-edd9-4021-a485-f4c09d64206d.png)
> First 20 data points as training data and last 10 as testing dataset

![image](https://user-images.githubusercontent.com/121909443/236701482-7adefdeb-804f-4777-8c0b-2c00f61d4cde.png)
> First 10 and last 10 data points as training data and last 10 as testing dataset 

We can see that when we are using first 10 and last 10 data points as training data and last 10 as testing dataset, training error is 12% lower and testing data is 3018% lower than the othercase. This is because neural network might be overfitting to the training data in the first case, which means that it is becoming too specialized to the training data and is not able to generalize well to new data. Since the first 20 data points might not be representative of the entire data set, the neural network might be learning specific features of this subset that are not present in the remaining 10 data points, leading to larger errors when tested on the latter.

Moving on, we need to **compare the models fit in homework 1 to neural network in this assignment**

To compare the models from homework one to neural netowrk in the previous task, we can use least square rror as a measure of their performance of the training and testing data. Recall that the least square error for each model on the 
____ questions, shouldn't the LSE be smaller when you fit it on neural network?


Lastly, after computing the first 20 PCA modes of the digit images and building a feed-forward neural network to classify the digits, we can compare the results of the neural network against LSTM, SVM and decision tree classifiers. 
![image](https://user-images.githubusercontent.com/121909443/236702129-d2d9e49e-5cbe-4c73-90e9-eeec9b6144e3.png)
(accuracy of SVM, decision tree, and LSTM model)

Here is a breif comparison of their accuracy on the testset using different classifiers and neural network. 
- Feedforward neural network: The accuracy of the three-layer feedforward neural network on the MNIST test set is 0.9729, which is a high accuracy compared to other classifiers.
- LSTM: The accuracy of the LSTM classifier on the MNIST test set is 0.9269, which is lower than the accuracy of the feedforward neural network.
- SVM: The accuracy of the SVM classifier on the MNIST test set is 0.7418, which is lower than the accuracy of the feedforward neural network and LSTM.
- Decision tree: The accuracy of the decision tree classifier on the MNIST test set is 0.4131, which is significantly lower than the accuracy of other classifiers.

Therefore, the feedforward neural network has the highest accuracy among the tested classifiers. However, it is worth noting that the performance of these classifiers can vary depending on the hyperparameters and specific implementation used.

## Summary and Conclusions
In this assignment, we applied three layer feedforward neural network to fit the given dataset of 31 points. We then trainied a feedforwar neural network on the MNIST dataset to classify digits and comapred the results with other classifiers including LSTM, SVM, and decision trees.

In the first part of the assignment, we found that using the first 10 and last 10 data as training and 10 data points as testing data led a lwer error in comparison to using the first 20 data points as training data and the remaining 10 data points as testing data.

Next, for the second part, we find the first 20 PCA modes of the digit images and build a feedforwar neural network. after comparing the results with other classifiers such as SVM, LSTM, and decision tree. We found that neural network achieved a relatively higher accuracy. It may perform better with further tuning and optimzation on the parameters. 

Overall, this assignment gives a good opportunity to apply neural networks and other classifier to real world dataset such as MNIST. We explore different models and compare their performance which allows us to visualize and think about the appropriate model. Also, using the PCA can really improve the performance of the classifiers. In conclusion, this assignment is a good practice on the neural network and their applications, as well as introduced us to other common classifiers in machine learning in general. 
