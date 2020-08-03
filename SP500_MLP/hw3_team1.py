#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:41:33 2019

@author: APP Prog for FE -- Team 1
"""
### import packages ###
import pandas as pd
import numpy as np
from sklearn import preprocessing

### read data ###
data_dir = "/Users/zihanwang/Documents/Columbia/2019Spring/Applications_Programming/hw/"
data = pd.read_csv(data_dir+'hw3index.csv')

### Pre-Process data ###
data = data.drop('Unnamed: 1', axis = 1)  # (Assets = 486, Dates = 746)
fill_method = 'lastfill'  # Parameter
if fill_method == 'lastfill':
    data = data.fillna(method='ffill')
elif fill_method == 'linear interpolate':
    data = data.interpolate()
elif fill_method == '10d MA':
    data = data.fillna(data.rolling(11, min_periods=1).mean())
# Missing data still possible for assets that publish later
data = data.fillna(method='bfill')
#data = data.fillna(data.mean())
# data normalization
scaler = preprocessing.StandardScaler().fit(data)
data = pd.DataFrame(scaler.transform(data), columns = data.columns)

y_data = data["('Close', 'SPX')"].values[10:]
X_data = data.drop("('Close', 'SPX')", axis = 1).T.values[:, :(data.shape[0]-10)] # (486, 746)
# X_data.shape = (486, 746)
# CHECK NA
check = np.sum(data.isnull().sum())
if check == 0:
    print('No NAs in data')
# data.shape = (756, 487)
y_data = y_data.reshape(((data.shape[0]-10),1))
T = int(data.shape[0] * 0.8)
X_train = X_data[:,0:T]
y_train = y_data[0:T]
X_test = X_data[:,T:(data.shape[0]-10)]
y_test = y_data[T:(data.shape[0]-10)]

print('X_train.shape', X_train.shape, ',y_train.shape', y_train.shape, \
      ',X_test.shape', X_test.shape, ',y_test.shape', y_test.shape,)

### Neural Network ###
class ReLuNeuralNetwork():
    
    def __init__(self, input_dim, hidden_nodes1, hidden_nodes2, hidden_nodes3, output_dim):
        # define NN architecture
        self.input_dim = input_dim
        self.hidden_nodes1 = hidden_nodes1
        self.hidden_nodes2 = hidden_nodes2
        self.hidden_nodes3 = hidden_nodes3
        self.output_dim = output_dim

        self.model = {} ## Now consider W/O bias term
        self.model['W1'] = np.random.normal(size =(self.input_dim, self.hidden_nodes1))
        #model['b1'] = np.zeros((1, hidden_nodes1))
        self.model['W2'] = np.random.normal(size =(self.hidden_nodes1, self.hidden_nodes2))
        #model['b2'] = np.zeros((1, hidden_nodes2))
        self.model['W3'] = np.random.normal(size =(self.hidden_nodes2, self.hidden_nodes3))
        #model['b3'] = np.zeros((1, hidden_nodes3))
        self.model['W4'] = np.random.normal(size =(self.hidden_nodes3, self.output_dim))
        print('Weights shapes:', self.model['W1'].shape,self.model['W2'].shape, \
              self.model['W3'].shape, self.model['W4'].shape )
        print('\nInitial Weights',(self.model['W1'].max(),self.model['W1'].min(),self.model['W1'].mean()), \
              (self.model['W2'].max(), self.model['W2'].min(), self.model['W2'].mean()), \
              (self.model['W3'].max(), self.model['W3'].min(), self.model['W3'].mean()), \
              (self.model['W4'].max(), self.model['W4'].min(), self.model['W4'].mean()))
    def relu(self,X):
        return np.maximum(X, 0)

    def relu_derivative(self,A): # da_J/dZ_J
        return 1. * (A > 0)

    def feed_forward(self, x):
        self.W1, self.W2, self.W3, self.W4= self.model['W1'], self.model['W2'], self.model['W3'], self.model['W4']
        # Forward propagation
        #x = x.reshape((x.shape[0],1))
        z1 = np.dot(self.W1.T, x)#x.dot(self.W1)
        a1 = self.relu(z1)
        z2 = np.dot(self.W2.T, a1)#a1.dot(self.W2)
        a2 = self.relu(z2)
        z3 = np.dot(self.W3.T, a2)#a2.dot(self.W3)
        a3 = self.relu(z3)
        z4 = np.dot(self.W4.T, a3)#a3.dot(self.W4)
        out = self.relu(z4)

        self.model['A0'] = x
        self.model['A1'] = a1
        self.model['A2'] = a2
        self.model['A3'] = a3
        self.model['A4'] = out

        print('FeedForward A.shape:', x.shape,a1.shape, a2.shape, a3.shape, out.shape )

        return z1, a1, z2, a2, z3, a3, z4, out
    
    def calculate_loss(self,X,y):
        self.W1, self.W2, self.W3, self.W4 = self.model['W1'], self.model['W2'], self.model['W3'], self.model['W4']
        # Forward propagation to calculate our predictions
        z1, a1, z2, a2, z3, a3, z4, out = self.feed_forward(X)
        y = np.matrix(y).reshape((1, y.shape[0]))
        print('out.shape',out.shape, 'y.shape', y.shape)
        # Calculate Loss:
        loss = 1/2 * np.sum(np.square(out - y))
        # Calculating the loss
        #loss = np.log(np.sum(np.square(out - y))/(len(X)**2))
        # Add regulatization term to loss (optional)

        return loss
    
    def backprop(self,X,y,z1,a1,z2,a2,z3,a3,z4,output):
        #delta4 = 2.0 * (output - y) #yhat - y
        print('X.shape', X.shape, 'y.shape', y.shape)
        # Layer 4
        delta4 = (output - y) * self.relu_derivative(output) # dLoss/dyhat, (1,1)
        dW4 = np.dot(a3, delta4.T) # Relu, dyhat / dW4, (125, 1)
        #print('delta4.shape', delta4.shape, '|dW4.shape',dW4.shape) # (125,1)

        # Layer 3
        delta3 = np.multiply(self.model['W4'].dot(delta4) ,self.relu_derivative(a3))
        # (125,1) * (1,1) #delta4.dot(self.model['W4'].T) * self.relu_derivative(a3)
        dW3 = np.dot(a2, delta3.T) # (250,1) *(1,125) (dW3 = np.dot(a2.T, delta3)
        #print('delta3.shape', delta3.shape, '|dW3.shape', dW3.shape)

        # Layer 2
        delta2 = np.multiply(self.model['W3'].dot(delta3) ,self.relu_derivative(a2))
            #delta3.dot(self.model['W3'].T) * self.relu_derivative(a3) #if ReLU
        dW2 = np.dot(a1, delta2.T) #dW2 = np.dot(self.relu_derivative(a1), delta2.T)#
        #print('delta2.shape', delta2.shape,'|dW2.shape', dW2.shape)

        # Layer 1
        delta1 = np.multiply(self.model['W2'].dot(delta2) ,self.relu_derivative(a1))
            #delta2.dot(self.model['W2'].T) * self.relu_derivative(a1) #if ReLU
        dW1 = np.dot(X, delta1.T) #dW1 = np.dot(X.T, delta1)
        #print('delta1.shape', delta1.shape, '|dW1.shape', dW1.shape)
        # Add regularization terms
        #dW4 += reg_lambda * model['W4']
        #dW3 += reg_lambda * model['W3']
        #dW2 += reg_lambda * model['W2']
        #dW1 += reg_lambda * model['W1']
        return dW1, dW2, dW3, dW4

    def train(self, X, y, num_passes=10000, learning_rate=1):
        # Batch gradient descent
        done = False
        previous_loss = float('inf')
        i = 0
        losses = []
        while done == False:  #comment out while performance testing
        #while i < 1500:
            #feed forward
            #One Epoch
            for t in range(X.shape[1]):
                print('%dth iter _ %dth sample' % (i, t+1))
                X_sample = np.matrix(X[:,t]).reshape((X[:,t].shape[0],1))
                y_sample = np.matrix(y)[t,:]
                z1,a1,z2,a2,z3,a3,z4,output = self.feed_forward(X_sample)

                #backpropagation
                dW1_temp, dW2_temp, dW3_temp, dW4_temp = self.backprop(X_sample,y_sample,z1,a1,z2,a2,z3,a3,z4,output)
                try:
                    dW1 += dW1_temp /X.shape[1]
                    dW2 += dW2_temp / X.shape[1]
                    dW3 += dW3_temp / X.shape[1]
                    dW4 += dW4_temp / X.shape[1]
                except:
                    dW1 = dW1_temp / X.shape[1]
                    dW2 = dW2_temp / X.shape[1]
                    dW3 = dW3_temp / X.shape[1]
                    dW4 = dW4_temp / X.shape[1]

                print('dW1_temp',np.count_nonzero(dW1_temp),\
                      'dW2_temp',np.count_nonzero(dW2_temp),\
                      'dW3_temp', np.count_nonzero(dW3_temp),\
                      'dW4_temp',np.count_nonzero(dW4_temp))

            #update weights and biases
            self.model['W1'] -= learning_rate * dW1
            self.model['W2'] -= learning_rate * dW2
            self.model['W3'] -= learning_rate * dW3
            self.model['W4'] -= learning_rate * dW4

            print('\nW1_n', np.count_nonzero(self.model['W1']), \
              '\nW2_n', np.count_nonzero(self.model['W2']), \
              '\nW3_n', np.count_nonzero(self.model['W3']), \
              '\nW4_n', np.count_nonzero(self.model['W4']))

            print('\nTrained_weights', (self.model['W1'].max(), self.model['W1'].min(), self.model['W1'].mean()), \
                  '\n', (self.model['W2'].max(), self.model['W2'].min(), self.model['W2'].mean()), \
                  '\n', (self.model['W3'].max(), self.model['W3'].min(), self.model['W3'].mean()), \
                  '\n',(self.model['W4'].max(), self.model['W4'].min(), self.model['W4'].mean()))
            # print('W1',self.model['W1'])
            # print('W2',self.model['W2'])
            # print('W3', self.model['W3'])
            # print('W4', self.model['W4'])

        #if i % 1 == 0:
            loss = self.calculate_loss(X, y)
            losses.append(loss)
            print("Loss after iteration %i: %f" %(i, loss))  #uncomment once testing finished, return mod val to 1000
            if (previous_loss-loss)/previous_loss < 1e-8:
                done = True
                print('\tEnd at %dth Epoch\t'%i)
            previous_loss = loss
            i += 1
        return self.model, losses
    
    def prediction(self,x):
        z1,a1,z2,a2,z3,a3,z4,output = self.feed_forward(x)
        return output
    
    def test(self, X_test, y_test):
        mpe_sum = 0
        for y, x in zip(y_test, X_test):
            x = np.matrix(x).reshape((x.shape[0],1))
            y = np.matrix(y).reshape((y.shape[0],1))
            prediction = self.prediction(x)
            mpe_sum += 1/2 * (y - prediction) ** 2
            #mpe_sum += abs((y - prediction)/y)
            mpe = mpe_sum / len(y)
        print(mpe)


Relu = ReLuNeuralNetwork(X_train.shape[0],400,250,125,1)
Relu.train(X_train,y_train)
# Relu.test(X_test,y_test)