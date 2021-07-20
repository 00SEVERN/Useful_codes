# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:32:34 2021

@author: csevern
"""
#%% Attempt 1 this only ran into an issue on one example which required a rewrite
def wordBreak(s,wordDict):
    slist = list(s)
    wordc = 0
    for w in wordDict:
        print(w,''.join(slist), len(s),wordc)
        if w in ''.join(slist):
            print(w,"in",''.join(slist), int(s.count(w)))
            #if (len(s)-(len(w)*s.count(w))) in lendict or len(s)==len(w):
            wordc += 1
            slist[slist.index(w[0]):slist.index(w[-1])+2] = " "
            
            print(len(slist),wordc,slist)
        if len(s)==wordc:
            return True
    return False


print(wordBreak(

"ccaccc",
["cc","ac"]))

#%% Attempt 1 this only ran into an issue on one example which required a rewrite
def wordBreak(s,wordDict):
    word = ""
    wordc = 0
    for i,l in enumerate(s):
        word = word + l
        if word in wordDict:
            wordc += 1
            s = s[:i+1].replace(word," ") + s[i+1:] 
            print(s)
            word = ""
        if len(s)==wordc:
            return True
    return False
        


print(wordBreak(
"ccaccc",
["cc","ac"]))

#%% Attempt 3 again another solution, online solution very clever
def wordBreak(s,wordDict):
    dp = [False]*(len(s)+1)
    dp[len(s)] = True

    
    for i in range(len(s)-1,-1,-1):
        for w in wordDict:
            if (i+len(w)) <= len(s) and s[i:i+len(w)]==w:
                dp[i] =dp[i+len(w)]

            if dp[i]:
                break
    return dp[0]
        


print(wordBreak(
"ccaccc",
["cc","ac"]))


