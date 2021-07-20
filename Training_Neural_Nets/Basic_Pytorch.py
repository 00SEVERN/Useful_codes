# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:19:26 2021

@author: csevern
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

X = torch.tensor(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=torch.float) 
y = torch.tensor(([0],[1],[0],[1]), dtype=torch.float) 
xPredicted = torch.tensor(([0,1,0]), dtype=torch.float) 

print(X.size())
print(y.size())

class Neural_Network(nn.Module):
    def __init__(self, X):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = X.shape[1]
        self.outputSize = 1
        self.hiddenSize = X.shape[0]
        
        #Torch.randn just creates random numbers in the array size you specify
        #Create an array that is your input array dimensions transposed
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) 
        
        #Create a mimic output array that is the same dimensions as real output y
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) 
        
    def forward(self, X):
        self.z = torch.matmul(X, self.W1) #Multiply the input by Weights 1
        self.z2 = self.sigmoid(self.z) # Run that through sigmoid function
        self.z3 = torch.matmul(self.z2, self.W2) # Multiply that by Weights 2
        o = self.sigmoid(self.z3) # Then sigmoid that again and return results
        return o
        
    def sigmoid(self, s):
        #Sigmoid going forward
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid, or as I call it sigmoid going backwards
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o # The Difference between the real answer and our predicted one
        self.o_delta = self.o_error * self.sigmoidPrime(o) # Difference x's the sigmoid backward of our output
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2)) #multiply the output by transposed weights 2
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) #multiply that output by sigmoid reverse of z2 
        self.W1 += torch.matmul(torch.t(X), self.z2_delta) #Alter weights one
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta) #Alter weights 2
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))
        
NN = Neural_Network(X)
for i in range(1000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)
NN.predict()



## https://pytorch.org/tutorials/beginner/nn_tutorial.html Next step