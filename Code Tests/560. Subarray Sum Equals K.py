# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:26:36 2021

@author: csevern
"""
#%% First attempt, timelimit exceeded

def subarraySum(nums, k):
    arrays = [(x,y) for x in range(0,len(nums)) for y in range(len(nums)-1,-1,-1) if y>=x]
    
    sums = [sum(nums[x[0]:x[1]+1]) for x in arrays if sum(nums[x[0]:x[1]+1])==k]
    return len(sums)    
    
print(subarraySum([1,2,3],3))

#%% attempt 2 faster but still times out

def subarraySum(nums, k):
    count =0
    for i in range(0,len(nums)):
        for j in range(i,len(nums)):
            if sum(nums[i:j+1]) == k:
                count+=1
    return count
        
    
print(subarraySum([1,2,3],3))

#%% Online solution watched very useful video
#https://www.youtube.com/watch?v=HbbYPQc-Oo4&ab_channel=TECHDOSE
def subarraySum(nums, k):
        count, cur, res = {0: 1}, 0, 0
        for v in nums:
            cur += v
            res += count.get(cur - k, 0)
            count[cur] = count.get(cur, 0) + 1
        return res
print(subarraySum([-1,-1,1],0))


