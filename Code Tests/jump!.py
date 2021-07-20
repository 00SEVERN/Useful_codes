# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:41:35 2021

@author: csevern
"""
#%% this one is so complicated first attempt is slow but successful, beats only 5% on speed and ram
def canJump(nums):
    first = nums[0]
    print(first)
    if first >= len(nums)-1:
        return True
    elif first == 0:
        return False
    else:
        jumps = [x+i for i, x in enumerate(nums)]
        print(jumps)
        for i,j in enumerate(jumps):
            if i > 0 and max(jumps[:i+1]) == i:
                print("stop one", max(jumps[:i]),i)
                return False
            if j >= len(nums)-1 and i <= max(jumps[:i]):
                return True
        return False
                
                    


print(canJump([1,2,3]))

#%% Second attempt is refined, faster than 97.82%, more ram efficient than 95.78%
def canJump(nums):
    maxj = 0
    for i,n in enumerate(nums):
        j=n+i
     
        if j > maxj:
            maxj =j
        print(j,i,n, maxj, len(nums)-1)
        if j == i and j< len(nums)-1 and j == maxj:
            return False
        if j >= len(nums)-1:
            return True
    return False
                
                    


print(canJump([0,1]))