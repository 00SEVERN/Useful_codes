# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:52:01 2021

@author: csevern
"""
import random
grid = [[random.randint(1,10)]*100 for x in range(0,100)]

#%% attempt1  faster than 38%, more ram efficient than 36%
import time
time_start = time.time()
def minPathSum(grid):
    n,m = len(grid), len(grid[0])
    path = [[0]*m for x in range(0,n)]
    for x in range(0,n):
        for y in range(0,m):
            if y > 0 and x > 0:
                path[x][y] = min(path[x-1][y]+grid[x][y], path[x][y-1]+grid[x][y])
            elif y> 0:
                path[x][y] = path[x][y-1]+grid[x][y]
            else:
                path[x][y] = path[x-1][y]+grid[x][y]
    
    
    return path[n-1][m-1]




#grid = [[1,3,1],[1,5,1],[4,2,1]]
print(minPathSum(grid), time.time()-time_start)

#%% attempt2  96.2% faster, 88.3% more ram efficient
import time
time_start = time.time()
def minPathSum(grid):
    n,m = len(grid), len(grid[0])
    path = [[0]*m for x in range(0,n)]
    path[0] = [sum(grid[0][:x+1]) for x in range(0,m)]
    for x in range(1,n):
        for y in range(0,m):
            if y > 0 and x > 0:
                path[x][y] = min(path[x-1][y]+grid[x][y], path[x][y-1]+grid[x][y])
            else:
                path[x][y] = path[x-1][y]+grid[x][y]
    
    return path[n-1][m-1]


#grid = [[9,1,4,8]]
print(minPathSum(grid),time.time()-time_start)

#%% attempt3
import time
time_start = time.time()
def minPathSum(grid):
    n,m = len(grid), len(grid[0])
    for x in range(0,n):
        for y in range(0,m):
            if y == 0 and x == 0:
                continue
            if y>0 and x > 0:
                grid[x][y] = min(grid[x-1][y]+grid[x][y], grid[x][y-1]+grid[x][y])
            elif y > 0:
                grid[x][y] = grid[x][y-1]+grid[x][y]
            else:
                grid[x][y] = grid[x-1][y]+grid[x][y]


    return grid[n-1][m-1]


#grid = [[9,1,4,8]]
print(minPathSum(grid),time.time()-time_start)

#%% attemp3 O(n)
import time
time_start = time.time()
def minPathSum(grid):
    n,m = len(grid), len(grid[0])
    for x in range(1,m):
        grid[0][x]+=grid[0][x-1]
    for y in range(1,n):
        grid[y][0]+=grid[y-1][0]
    for x in range(1,n):
        for y in range(1,m):
            grid[x][y] = min(grid[x-1][y]+grid[x][y], grid[x][y-1]+grid[x][y])

    return grid[n-1][m-1]


#grid = [[9,1,4,8]]
print(minPathSum([[1,3,1],[1,5,1],[4,2,1]]),time.time()-time_start)