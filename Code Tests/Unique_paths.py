# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:07:32 2021

@author: csevern
"""
#%% basic version where you just fill a grid, consult. Horribly innefficient 
#https://researchideas.ca/wmt/c6b3.html
import numpy as np
def uniquePaths(m,n):
    grid = np.zeros([m, n], dtype=int)+1
    for x in range(1,m):
        for y in range(1,n):
            grid[x][y] = grid[x][y-1] + grid[x-1][y]
    print(grid)
    return grid[m-1,n-1];
    
print(uniquePaths(3,7))

#%% this is much faster, not using numpy, 59% faster, 63% more ram efficient
#https://researchideas.ca/wmt/c6b3.html
def uniquePaths(m,n):
    grid = [[1]*n for i in range(m)]
    for x in range(1,m):
        for y in range(1,n):
            grid[x][y] = grid[x][y-1] + grid[x-1][y]
    return grid[m-1][n-1];
    
print(uniquePaths(3,7))

#%% had to look up the factorial equation for this since I knew there was one, writing your own factorial code is faster
#than using math.factorial, still is 84% faster, 38% ram efficient. ALways a tradeoff

#https://researchideas.ca/wmt/c6b3.html

def fact(i):
    sum1 = 1
    for k in range(1,i+1):
        sum1 = sum1*k
    return sum1;
def uniquePaths(m,n):
    return int(fact(m+n-2)/(fact(m-1)*fact(n-1)))
    
print(uniquePaths(3,7))