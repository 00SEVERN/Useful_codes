# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:19:11 2021

@author: csevern
"""
#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
#Read above article, it explains the maths behind it, this code will explain the code aspects

import numpy as np
import math
def sigmoid(x):
    return 1 / (1 + math.e ** -x)

def sigmoid_derivative(p):
    return p * (1-p)

#Try to learn that the first play position is what determines what y is
#So if the first number is 0, the y will be y. Feel free to change the numbers
#And the size of the array, however always make sure the output is 1D
#Not also you define the dtype as float, so that even though you're only using
#integers, when you multiply the array by a np.random array (which is floats)
#you dont get an error

X = np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y = np.array(([0],[1],[0],[1]))


class NeuralNetwork:
    def __init__(self,x,y):
        #Define the input array
        self.input = x
        
        #Create an array that is your input array dimensions transposed
        #In this example a 4,3 shape goes toa  3,4 shape
        #This array is full of random numbers
        self.weights1 = np.random.rand(self.input.shape[1],self.input.shape[0])

        #Create a mimic output array that is the same dimensions as real output y
        #Again fill it with random numbers 
        self.weights2 = np.random.rand(self.input.shape[0],1)
        self.y = y

        #Define the output as an array of 0s matching the real output size
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        #Using the sigmoid function, multiply the input array by weights 1
        #This is your first layer, it creats a 4 x 4 (3x4 . 4x3)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))

        #Using the sigmoid function, multiply the first layer, by weights2
        #This is your output layer, it creates a 1x4 just like real y output
        
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))

        return self.layer2
    
    def backprop(self):
        #Backpropagation works out how close your output was to your input, and how do your
        #layers and weights need to be changed, remember your weights were initially
        #just random numbers, now they are being alterted to get closer to what real weights
        #they should be
        
        #To reverse the matrix multiplications, you must transpose the arrays, that is what .T is
        #Using sigmoid derivative (i.e. reverse sigmoid), work out what the difference between
        #Real output y and fake output y (which is layer 2 in feedfoward)
        #Start from going ouput(layer2) -> layer1
        d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output)*sigmoid_derivative(self.output))
       
        #You have to reverse the weights in the layers that they are done
        #Both layers include the difference between the output and real, as that is how you
        #Work out how much change needs to be made
        #So do the same as above, layer1 -> input
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y - self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))

        #Alter the weights by the newly calculated weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
    def train(self, X, y):
        #Train by making your output the output of your feedforward layer
        self.output = self.feedforward()
        
        #Then run the backpropogation so that next iteration the weights are changed
        #So that your fake output is closer to your real output
        self.backprop()
    def test(self,X):
        self.input = X
        self.output = self.feedforward()
        return  self.output
        
NN = NeuralNetwork(X,y)
test_X = np.array(([0,1,0]), dtype=float)
#Train it 2000 times
for i in range(1500):

    if i % 100 == 0:
        print("for iteration #" + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y-NN.feedforward()))))
        print("\n")
    NN.train(X,y)
print(str(NN.test(test_X)))    

    