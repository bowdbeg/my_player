import subprocess
import argparse
import sys
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import traceback
import time
import platform
import random

TIMEOUT_TIME = 3
DRAW = 255
PIECES = 3
#
BOARD_SIZE = 3
BOARD_LEN = BOARD_SIZE**3

# module for communicate with player
# https://qiita.com/kei0425/items/69fe513caab654a00e73


def board23d(board, num):
    board3d = board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
    b = (board3d == num).astype(int)
    piece = b.sum((0, 1))
    return board3d, piece


def forward(board, order, player_idx):
    if order[0] == -1:
        return board
    board3d = board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
    if board3d[order[0], order[1], order[2]] == 0:
        board3d[order[0], order[1], order[2]] = player_idx
    else:
        assert 'can not forward board, fix valid_move'

    return board


def valid_move(board, num):
    board3d, piece = board23d(board, num)
    av_piece = piece < PIECES
    valid = []
    for i, av in enumerate(av_piece):
        if av:
            b = board3d[:, :, i]
            for j, row in enumerate(b):
                for k, pix in enumerate(row):
                    if pix == 0:
                        valid.append([j, k, i])
    return valid


def valid_str(board, num):
    valid = valid_move(board, num)
    val = []
    for v in valid:
        val.append(' '.join(map(str, v)))
    return val


def calc_score(index, res, two_flag=False):
    score = np.zeros(len(index))
    if res == DRAW:
        score.fill(1 / len(index))
    elif res > 0:
        score[index[res - 1]] = 1
    elif res < 0:
        if two_flag:
            score.fill(0.5)
            score[index[abs(res) - 1]] = 0
            score[((index[abs(res) - 1]) + 2) % 4] = 0
        else:
            score.fill(1 / (len(index) - 1))
            score[index[abs(res) - 1]] = 0
    return score


def board2str(board):
    sboard = list(map(lambda x: str(int(x)), board.tolist()))
    return ' '.join(sboard)


def parse_player(arg):
    a = arg.split()
    if len(a) < 1:
        return '', []
    player = a[0]
    if len(a) > 1:
        player_arg = a[1:]
    else:
        player_arg = []
    return player, player_arg


def hand1d23d(h):
    return (h // 9, (h % 9) // 3, h % 3)


def judge(board):
    board3d = board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
    max_idx = board3d.max()
    res = 0
    for idx in range(1, int(max_idx) + 1):
        faces = []
        f1 = []
        f2 = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if all(board3d[i, j, :] == idx) or all(
                        board3d[:, i, j] == idx) or all(
                            board3d[i, :, j] == idx):
                    # tate yoko
                    assert not (res != 0
                                and res != idx), 'wrong judge tate yoko'
                    res = idx
                f1.append(board3d[i, j, i])
                f2.append(board3d[2 - i, j, i])
            faces.append(board3d[i, :, :])
            faces.append(board3d[:, i, :])
            faces.append(board3d[:, :, i])
        faces.append(np.array(f1).reshape([BOARD_SIZE, BOARD_SIZE]))
        faces.append(np.array(f2).reshape([BOARD_SIZE, BOARD_SIZE]))
        for face in faces:
            if all(np.diag(face) == idx) or all(
                    np.diag(np.fliplr(face)) == idx):
                # across
                assert not (res != 0 and res != idx), 'wrong judge across'
                res = idx
    if res == 0:
        val_len = [len(valid_move(board, i)) == 0 for i in range(4)]
        if all(val_len):
            res = DRAW
    return res

def index2line(index):
    return index[0]*9+index[1]*3+index[2]

# player have to callable like
# player(board)
# return legal hand
# game not judge whether legal or ilegal
def game(player1, player2):
    board = np.array([0 for i in range(27)])
    turn = 0
    while True:
        # print(board)
        player_num = turn % 4 + 1
        player = [player2, player1][player_num % 2]
        hand = player([player_num, *board])
        if all([h == -1 for h in hand]):
            turn += 1
            continue
        board[index2line(hand)] = player_num
        jge = judge(board)
        if jge != 0:
            break
        turn += 1
    if jge == DRAW:
        jge = -1
    else:
        jge = jge % 2
    return jge


def random_player(msg):
    my_num = msg[0]
    board = np.array(msg[1:])
    val = valid_move(board, my_num)
    if not val:
        return [-1,-1,-1]
    move = random.choice(val)
    return move
