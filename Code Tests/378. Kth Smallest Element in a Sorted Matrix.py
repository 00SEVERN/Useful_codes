# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:57:46 2021

@author: csevern
"""
#%% first attempt, faster than 91.16%, more ram efficient than 78.08%

def kthSmallest(matrix, k):    
       
    return [y for x in matrix for y in x][k-1];

print(kthSmallest([[1,5,9],[10,11,13],[12,13,15]],8))