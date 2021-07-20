# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:35:24 2021

@author: csevern
"""
#%% attempt one faster than 96%, more ram efficient than 79.62%
def searchRange(nums, target):
    if len(nums)>0 and target in nums:
        return list([nums.index(target),len(nums)-nums[::-1].index(target)-1])
    elif len(nums)==1 and target in nums:
        return list([0,0])
    else:
        return list([-1,-1])
    
print(searchRange([5,7,7,8,8,10],8))