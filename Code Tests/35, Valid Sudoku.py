# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:46:02 2021

@author: csevern
"""
#%% First try faster than 98%, more Ram efficient than 68%
def isValidSudoku(board):
    def rowcheck(board):
        for i in board:
            if len(list(set(i)))!= 9-(i.count(".")-1):
                return False
        return True
    
    def columncheck(board):
        for i in range(0,9):
            col = [board[x][i] for x in range(0,9)]
            if len(list(set(col)))!= 9-(col.count(".")-1):
                return False          
        return True

    def squarecheck(board):
        squares = [x for x in range(3,10,3)]
        prev1 = 0
        for q in squares:
            prev = 0
            for v in squares:
                sq = []
                square = [sq.extend(x[prev:v]) for x in board[prev1:q]]
                if len(list(set(sq)) )!= 9-(sq.count(".")-1):
                    return False 
                prev = v
            prev1 = q
        return True
    
    if rowcheck(board) == True:
        if columncheck(board) == True:
            if squarecheck(board) == True:
                return True
            else:
                return False
        else:
            return False
    else:
        return False
        
        


suduko = [["5","3",".",".","7",".",".",".","."],
          ["6",".",".","1","9","5",".",".","."],
          [".","9","8",".",".",".",".","6","."],
          ["8",".",".",".","6",".",".",".","3"],
          ["4",".",".","8",".","3",".",".","1"],
          ["7",".",".",".","2",".",".",".","6"],
          [".","6",".",".",".",".","2","8","."],
          [".",".",".","4","1","9",".",".","5"],
          [".",".",".",".","8",".",".","7","9"]]

print(isValidSudoku(suduko))