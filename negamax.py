import random
import numpy as np
import time
from utils import *

PIECES = 3
BOARD_SIZE=3

# player for test
# say minimum index of valid move

get = list(map(int, input().strip().split(' ')))
player_num = get[0]
my_turn=get[1:]
print(' '.join(map(str,my_turn)))

board = np.array(list(map(int, input().strip().split(' '))))
piece = np.zeros(3)

print(0)


def get_row(board3d):
    row=[]
    for i in range(board3d.shape[0]):
        for j in range(baord3d.shape[1]):
            row.append(baord3d[i,j,:])
            row.append(baord3d[i,:,j])
            row.append(baord3d[:,i,j])
        row.append(baord3d[i,i,:])
        row.append(baord3d[:,i,i])
        row.append(baord3d[i,:,i])
    l=len(board3d)-1
    row.append(np.array([board3d[i,i,i] for i in range(len(board3d))]))
    row.append(np.array([board3d[l-i,i,i] for i in range(len(board3d))]))
    row.append(np.array([board3d[l-i,i,l-i] for i in range(len(board3d))]))
    row.append(np.array([board3d[i,i,l-i] for i in range(len(board3d))]))

    
def get_score_rule(board,player):
    score=0
    board3d = board23d(board)
    WIN=100

    rows=get_row(board3d)
    rows
    return score


def get_score(board,player):
    return get_score_rule(board,player)

def alpha_beta_r(board,alpha,beta,player,depth):
    if depth==0:
        return get_score(board,player) , None
    best_score = -float('inf')
    best_move=None
    moves = valid_move(board,player)
    for move in moves:
        board_f=forward(board,move,player)
        score, child_best = alpha_beta_r(board_f, -beta, -alpha, depth-1)
        score = -score
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break
    return best_score, best_move

def alpha_beta(board, player, depth):
    return alpha_beta_r(board, -float('inf'), float('inf'), depth)



# main
if __name__=='__main__':
    try:
        while True:
            get = np.array(list(map(int, input().strip().split(' '))))
            turn=get[0]
            board=get[1:]
            valid=valid_move(board,turn)
            if len(valid) == 0:
                print(' '.join(['-1']*3))
                continue
            move = valid[0]
            print(' '.join(map(str,move)))
    except EOFError:
        pass