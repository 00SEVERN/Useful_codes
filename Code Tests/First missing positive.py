# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:44:02 2021

@author: csevern
"""
#%% First attempt 1, works but like 17% faster, and 15% ram efficient
def firstMissingPositive(nums):
    numss = sorted(nums)
    smallest = 10**100
    if 1 not in numss:
        return 1
    for i,x in enumerate(numss[1:]):
        value = abs(x-numss[i])
        if value % 1 == 0 and value >= 2 and numss[i]+1 < smallest and value != x and numss[i]>0:
            smallest = numss[i]+1
    if smallest == 10**100:
        smallest = numss[-1]+1

    return smallest
        
    #missing = [int((numss[-i]+x/2)) for i,x in enumerate(numss[1:]) if (numss[-i]+x/2)%1 == 0 and abs((numss[-i]+x/2))==(numss[-i]+x/2)]

    
    
print(firstMissingPositive([3,4,-1,1]))

#%% Second attempt better, but still not above 50%

def firstMissingPositive(nums):
    nums = sorted(list(set(nums)))
    print(nums)
    if 1 not in nums:
        return 1
    for i,n in enumerate(nums[:-1]):
        print(n, nums[i+1])
        if n+1 != nums[i+1] and n+1 >0:
            return n+1
    return nums[-1]+1
print(firstMissingPositive([3,4,-1,1]))

#%% third attempt kinda abandoning it, despite being good mathematical theory, too complex

def firstMissingPositive(nums):
    if 1 not in nums:
        return 1
    else:
        gauss = (max(nums)/2)*(min(nums)+max(nums))
        print(gauss, max(nums),min(nums),sum(nums))
        if int(gauss - sum(nums)) <= 0:
            return max(nums)+1
        else:
            if nums[-1]> len(nums):
                return min(nums)+1
            else:
                return int(gauss - sum(nums))
print(firstMissingPositive([1,2,3,4,5,6,7,8,9,20]))


#%% attempt 5
def firstMissingPositive(nums):
    nums.sort()
    if 1 in nums:
        find1 = nums.index(1)
    for i in range(1,max(nums)+1):
        if i != nums[i+find1-1]:
            return i
    return max(nums)+1
print(firstMissingPositive([1,2,3,4,5,6,7,8,9,20]))        