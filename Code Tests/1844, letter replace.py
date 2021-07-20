# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:07:09 2021

@author: csevern
"""
#%% try 1
def replacingDigits(s):
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    

    output = ''.join([str(s[n-1])+alpha[int(alpha.index(str(s[n-1])))+int(x)] for n,x in enumerate(s) if str(x).isdigit()==True])
    if len(output) == len(s)-1:
        output =output+(s[-1])
    
 
    return output;
print(replacingDigits("a1b2c3d4e"))
    
    
#%% try 2
import string
def replacingDigits(s):
    string.ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'

    output = ''.join([str(s[n-1])+string.ascii_lowercase[int(string.ascii_lowercase.index(str(s[n-1])))+int(x)] for n,x in enumerate(s) if str(x).isdigit()==True])
    if len(output) == len(s)-1:
        output =output+(s[-1])
    
 
    return output;
print(replacingDigits("a1b2c3d4e"))

#%% try 3 
def replacingDigits(s):
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    output=[]
    for n,x in enumerate(s):
        if x.isdigit()==True:
            output.append(alpha[int(alpha.index(str(s[n-1])))+int(x)])
        else:
            output.append(x)
 
    return ''.join(output);
print(replacingDigits("a1b2c3d4e"))

#%% try 4
def replacingDigits(s):
    s= list(s)
    for n in range(1,len(s),2):
        x=s[n]
        s[n]=chr(ord(s[n-1])+int(x))

    return ''.join(s);
print(replacingDigits("a1b2c3d4e"))    
        
#%% try 5
def replacingDigits(s):
    return ''.join([chr(ord(s[n-1])+int(x)) if x.isdigit()==True else x  for n,x in enumerate(list(s))]);
print(replacingDigits("a1b2c3d4e"))        
