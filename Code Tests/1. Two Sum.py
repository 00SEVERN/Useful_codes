# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:57:10 2021

@author: csevern
"""
#%% attempt 1 19.69% faster, 52.91% more efficient


def twoSum(nums, target):
    for i,n in enumerate(nums):
        misval = target-n
        nums2 = nums[i+1:]
        if misval in nums2:
            for k,v in enumerate(nums):
                if v==misval and k != i:
                    return [i,k]
    
    
    
    
print(twoSum([2,7,11,15],9))

#%% attempt 2


def twoSum(nums, target):
    for i,n in enumerate(nums):
        misval = target-n
        if misval in nums[i+1:]:
            return [i,nums[i+1:].index(misval)+i+1]
    
    
    
print(twoSum([3,3],6))

#%% attemp 3
def twoSum(nums,target):
    diffMap={}
    for i,n in enumerate(nums): 
        try:
            return [diffMap[target-n],i]
        except:
            diffMap[n] = i
                