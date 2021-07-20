# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:42:46 2021

@author: csevern
"""
#%% first attempt

def findcoin(coins, amount):
    sc = sorted(coins, reverse=True)
    am = amount
    cs = 0
    if amount == 0:
        return 0
    if (len(coins)==1 and amount % coins[0] != 0):
        return -1
    if min(coins)>amount:
        return -1
    for s in sc:
        if s <= am:
            if am % s == 0:

                cs += am//s
                am = 0
            else:
                n = am//s
                am= am-n*s
                cs+=n
                print(cs,am)
        if am == 0:
            break
    if am != 0:
        cs = -1
    return cs

print(findcoin([1],2))
            
            
                
        