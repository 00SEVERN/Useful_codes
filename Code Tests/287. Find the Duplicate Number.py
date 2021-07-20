# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:37:12 2021

@author: csevern
"""
import random
nums = open('L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Internal Coding/Python/Caleb/Training/Code Tests/long_num.txt', encoding='utf-8').read()
nums = nums.split(",")
#%% attempt1 slow and inefficient
import time
start_time = time.time()
import collections
def findDuplicate(nums):
    counts = collections.Counter(nums)
    return counts.most_common(1)[0][0]

print(findDuplicate(nums), time.time()-start_time)

#%% attempt2 without packages
import time
start_time = time.time()
def findDuplicate(nums):
    for i in range(0, len(nums)):
        num1 = nums[i]
        for j in range(len(nums)-1,i,-1):
            if num1 == nums[j]:
                return num1
            
print(findDuplicate(nums), time.time()-start_time)    

#%% attemp3 just list
import time
start_time = time.time()
def findDuplicate(nums):
    there = []
    for x in nums:
        if x in there:
            return x
        else:
            there.append(x)
        
            
print(findDuplicate(nums), time.time()-start_time)   


#%% attempt 4
import time
start_time = time.time()
def findDuplicate(nums):
    for i,x in enumerate(nums):
        if nums[i:].count(x)>1:
            return x
        
            
print(findDuplicate(nums), time.time()-start_time)   

#%% attempt 5 this faster and better than all previous versions so far
import time
start_time = time.time()
def findDuplicate(nums):
    nums = sorted(nums, reverse=True)
    print(nums)
    for i in range(1, len(nums)-1,3):
        print(nums[i-1],nums[i],nums[i+1])
        if nums[i]==nums[i+1] or nums[i]==nums[i-1]:
            return nums[i]
        
            
print(findDuplicate([1,3,4,2,2]), time.time()-start_time)   

#%% looking at solution they want you to use tortoise and hare, doesnt work in spyder, does work in leetcode. 
#I'm letting the cynic out here, but it seems this question was written purely for a specific usecase
#Basically some gatekeeping data scientist wrote a specific test to prove hes the best at doing linked lists
#But put it in normal algorithms
import time
start_time = time.time()
def findDuplicate(nums):
        tortoise = hare = int(nums[0])
        while True:

            tortoise = int(nums[int(tortoise)])
            hare = int(nums[int(nums[int(hare)])])
            if tortoise == hare:
                break

        tortoise = nums[0]
        while tortoise != hare:
            tortoise = nums[tortoise]
            hare = nums[hare]
        
        return hare

print(findDuplicate(nums), time.time()-start_time)   
