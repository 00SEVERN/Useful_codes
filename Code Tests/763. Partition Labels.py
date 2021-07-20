# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:40:48 2021

@author: csevern
"""
#%% First attempt slow but very ram efficient
import time
start_time = time.time()
def partitionLabels(s):
    intlist = []
    i=0
    j=i
    while i <= len(s)-1 and j< len(s):
        for j in range(i,len(s)):
            ul = list(set(s[i:j+1]))
            fo = len([x for x in ul if (x in s[0:i]) or (x in s[j+1:])])
            
            if fo == 0 or j==len(s)-1:
                intlist.append(len(s[i:j+1]))
                i=j
                break
        i+=1
      
    return intlist;

print(partitionLabels("caedbdedda"), time.time()-start_time)


#%% attempt 2 faster and slightly more ram efficient, 11.39% on speed, 93.79% on Ram
import time
start_time = time.time()
def partitionLabels(s):
        intlist = []
        i=0
        j=i
        while i <= len(s)-1 and j< len(s):
            for j in range(i,len(s)):
                fo=0
                for x in s[i:j+1]:
                    if (x in s[0:i]) or (x in s[j+1:]):
                        fo =1
                        break
                
                if fo == 0 or j==len(s)-1:
                    intlist.append(len(s[i:j+1]))
                    i=j
                    break
            i+=1
        return intlist;
    
print(partitionLabels("ababcbacadefegdehijhklij"), time.time()-start_time)

#%% attempt 3

import time
start_time = time.time()
def partitionLabels(s):
        intlist = []
        sb = ''.join(list(s)[::-1])
        ldict = {}
        ul = list(set(s))
        for u in ul:
            print(u,s.index(u),len(s)-sb.index(u) )
            ldict[u]=s[s.index(u):len(s)-sb.index(u)]
        print(s,sb)
        print(ldict)
        return intlist;
    
print(partitionLabels("ababcbacadefegdehijhklij"), time.time()-start_time)

