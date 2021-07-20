# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:14:39 2021

@author: csevern
"""
#%% attempt 1 faster but less ram efficient
def reverseVowels(s):
    sl = list(s)
    vowels = "aeiouAEIOU"
    wv = {x:v for x,v in enumerate(sl) if v in vowels}
    locations = list(reversed(wv.keys()))
    for i,l in enumerate(wv.values()):
        sl[locations[i]] = l
    return ''.join(sl)
        
print(reverseVowels("hello"))

#%% attempt 2 better in everyway annoyingly
def reverseVowels(s):
    i, j = 0, len(s) - 1
    vowels = "aeiouAEIOU"
    s = list(s)
    
    while i < j:
        if s[i] in vowels and s[j] not in vowels:
            while s[j] not in vowels:
                j -= 1
            s[i], s[j] = s[j], s[i]
        elif s[i] not in vowels and s[j] in vowels:
            while s[i] not in vowels:
                i += 1
            s[i], s[j] = s[j], s[i]
        elif s[i] in vowels and s[j] in vowels:
            s[i], s[j] = s[j], s[i]
        i += 1
        j -= 1    
    return "".join(s)

#%% perfect square

def isPerfectSquare(n):
    if n**0.5 % 1 == 0.0:
        return True
    else:
        return False

print(isPerfectSquare(16))