# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:57:15 2020

@author: csevern
"""
#%% Attempt 1
import numpy as np
def dld(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    print(d)

    for i in range(lenstr1):
        
        for j in range(lenstr2):
            
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1

            
            
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )

            print(i,j,s1[i],s2[j], d[(i,j)])             
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
    print(d)
    return d[lenstr1-1,lenstr2-1]


def dice(s1,s2):
    s1 = set(s1)
    s2 = set(s2)
    c1= len([x for x in s1 if x in s2])
    c2= len([x for x in s2 if x in s1])
    c = (c1+c2)/2
    dc = (2*c)/(len(s1) + len(s2))
    return dc

s1 = "filipe"
s2 = "phillip"

print(dld(s1,s2))
print(dice(s1,s2))

#%% Attempt 2

def dld(word1, word2):
    m = len(word1)
    n = len(word2)
    
    DParray = [[0 for j in range(n+1)] for i in range(m+1)]
    print(DParray)
    for i in range(m+1):
        for j in range(n+1):
            
            if i == 0:
                DParray[i][j] = j
                
            elif j == 0:
                DParray[i][j] = i
                
            elif word1[i-1] == word2[j-1]:
                DParray[i][j] = DParray[i-1][j-1]
                
            else:
                  DParray[i][j] = 1 + min(DParray[i][j - 1],  # Insert
                  DParray[i - 1][j],  # Remove
                  DParray[i - 1][j - 1])  # Replace
    print(DParray)
    return(DParray[i][j])

s1 = "filipe"
s2 = "phillip"

print(dld(s1,s2))