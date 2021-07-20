# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:45:37 2021

@author: csevern
"""
#%% attempt 1 taken from the internet because im weak. 

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode):
        if not root1 and not root2: return None
        ans = TreeNode((root1.val if root1 else 0) + (root2.val if root2 else 0))
        ans.left = self.mergeTrees(root1 and root1.left, root2 and root2.left)
        ans.right = self.mergeTrees(root1 and root1.right, root2 and root2.right)
        return ans
            
            
