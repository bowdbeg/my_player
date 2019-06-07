import subprocess
import argparse
import sys
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import traceback
import time
from timeout import timeout, TimeoutError
import platform

TIMEOUT_TIME = 3
DRAW = 255
PIECES = 3
#
BOARD_SIZE = 3
BOARD_LEN = BOARD_SIZE**3

# module for communicate with player
# https://qiita.com/kei0425/items/69fe513caab654a00e73


class player_pipe(object):
    def __init__(self, *args_1, **args_2):
        if 'encoding' in args_2:
            self.encoding = args_2.pop('encoding')
        else:
            self.encoding = 'utf-8'
        self.popen = subprocess.Popen(
            *args_1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding=self.encoding, **args_2)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def send(self, message, recieve=True, incr=False, verbose=False):
        message = message.rstrip('\n')
        if not incr and '\n' in message:
            raise ValueError("message in \\n!")
        self.popen.stdin.write(message + '\n')
        self.popen.stdin.flush()
        if recieve:
            return self.recieve()
        return None

    @timeout(TIMEOUT_TIME)
    def recieve(self):
        self.popen.stdout.flush()
        return self.popen.stdout.readline()

    def close(self):
        self.popen.kill()


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


def judge(board, players):
    board3d = board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
    max_idx = board3d.max()
    res = 0
    for idx in range(1, int(max_idx)+1):
        faces = []
        f1 = []
        f2 = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if all(board3d[i, j, :] == idx) or all(board3d[:, i, j] == idx) or all(board3d[i, :, j] == idx):
                    # tate yoko
                    assert not(res != 0 and res !=
                               idx), 'wrong judge tate yoko'
                    res = idx
                f1.append(board3d[i, j, i])
                f2.append(board3d[2-i, j, i])
            faces.append(board3d[i, :, :])
            faces.append(board3d[:, i, :])
            faces.append(board3d[:, :, i])
        faces.append(np.array(f1).reshape([BOARD_SIZE, BOARD_SIZE]))
        faces.append(np.array(f2).reshape([BOARD_SIZE, BOARD_SIZE]))
        for face in faces:
            if all(np.diag(face) == idx) or all(np.diag(np.fliplr(face)) == idx):
                # across
                assert not(res != 0 and res != idx), 'wrong judge across'
                res = idx
    if res == 0:
        val_len = [len(valid_move(board, i)) == 0 for i in range(players)]
        if all(val_len):
            res = DRAW
    return res


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


def calc_score(index, res,two_flag=False):
    score = np.zeros(len(index))
    if res == DRAW:
        score.fill(1/len(index))
    elif res > 0:
        score[index[res-1]] = 1
    elif res < 0:
        if two_flag:
            score.fill(0.5)
            score[index[abs(res)-1]]=0
            score[((index[abs(res)-1])+2)%4]=0
        else:
            score.fill(1/(len(index)-1))
            score[index[abs(res)-1]] = 0
    return score


def board2str(board):
    sboard = list(map(lambda x: str(int(x)), board.tolist()))
    return ' '.join(sboard)


def parse_player(arg):
    a=arg.split()
    if len(a) < 1:
        return '',[]
    player=a[0]
    if len(a) > 1:
        player_arg=a[1:]
    else:
        player_arg=[]
    return player,player_arg


########################################################################################
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-1', '--first', required=False, type=str, default='')
    parser.add_argument('-2', '--second', required=False, type=str, default='')
    parser.add_argument('-3', '--third', required=False, type=str, default='')
    parser.add_argument('-4', '--fourth', required=False, type=str, default='')
    parser.add_argument('--graphical', help='print score glaphically',
                        required=False, action='store_true')
    # https://blog.amedama.jp/entry/2018/07/13/001155
    parser.add_argument('--mode', help='once: play once   combination: play all combination', required=False,
                        choices=['once', 'combination'], default='once')
    parser.add_argument('--debug', help='when this switch is on, invalid operation will be not allowed.',
                        required=False, action='store_true')
    parser.add_argument('-o', '--output', help='log file path',
                        required=False, default='log.out', type=str)
    parser.add_argument(
        '-v', '--verbose', help='verbose game board and user output', action='store_true')
    parser.add_argument('-t','--times',help='battle t times',type=int,default=1)
    args = parser.parse_args()
    verbose = args.verbose

    players = []
    player_args=[]

    ply,plya = parse_player(args.first)
    players.append(ply)
    player_args.append(plya)

    ply,plya = parse_player(args.second)
    players.append(ply)
    player_args.append(plya)
    
    ply,plya = parse_player(args.third)
    players.append(ply)
    player_args.append(plya)
    
    ply,plya = parse_player(args.fourth)
    players.append(ply)
    player_args.append(plya)
    times=args.times
    Index = []
    Players = []
    for i, p in enumerate(players):
        if '.py' in p:
            Index.append(i)
            Players.append(p)

    if len(Index) < 2:
        raise ValueError('not enough player')

    if Index != list(range(len(Index))):
        raise ValueError('player order is wrong')

    scores = np.zeros(len(Index))
    two_flag = False

    if len(Index) == 2:
        two_flag = True
        if args.mode == 'once':
            indexes=[[0,1,0,1]]
        elif args.mode == 'combination':
            indexes = [[0, 1, 0, 1], [1, 0, 1, 0]]
    else:
        if args.mode == 'combination':
            indexes = itertools.permutations(Index)
        elif args.mode == 'once':
            indexes = [Index]
    with open(args.output, mode='w') as f:
        for _ in range(times):
            for index in indexes:
                try:
                    if verbose:
                        print('Initializing...')
                    f.write(' '.join(map(str, index))+'\n')
                    board = np.zeros(BOARD_LEN)

                    player_p = []

                    # connect pipe
                    if two_flag:
                        cmd0=['python3',players[index[0]]]
                        cmd1=['python3',players[index[1]]]
                        cmd0.extend(player_args[index[0]])
                        cmd1.extend(player_args[index[1]])
                        player0 = player_pipe(cmd0)
                        player1 = player_pipe(cmd1)
                        player_p = [player0, player1, player0, player1]

                        index_ = [index[0], index[1]]
                        for i, ind in enumerate(index_):
                            # send order
                            msg = '{} {} {}'.format(2, i+1, i+3)
                            if verbose:
                                print('[send : {}] : {}'.format(players[ind], msg))

                            rep = player_p[i].send(msg).strip()
                            if verbose:
                                print('[recv : {}] : {}'.format(players[ind], rep))
                            if rep != '{} {}'.format(i+1, i+3):
                                raise ValueError(
                                    '{} reply invalid value. (move)'.format(players[ind]))

                            # send board
                            msg = '{} {}'.format(1, board2str(board))
                            if verbose:
                                print('[send : {}] : {}'.format(players[ind], msg))

                            rep = player_p[i].send(msg).strip()
                            if verbose:
                                print('[recv : {}] : {}'.format(players[ind], rep))
                            if rep != str(0):
                                raise ValueError(
                                    '{} reply invalid value. (board Ping)'.format(players[ind]))
                    else:
                        for i, ind in enumerate(index):
                            cmd=['python3',players[ind]]
                            cmd.extend(player_args[ind])
                            player_p.append(player_pipe(cmd))
                            # send order
                            msg = '{} {}'.format(len(index), i+1)
                            if verbose:
                                print('[send : {}] : {}'.format(players[ind], msg))

                            rep = player_p[i].send(msg).strip()
                            if verbose:
                                print('[recv : {}] : {}'.format(players[ind], rep))
                            if rep != str(i+1):
                                raise ValueError(
                                    '{} reply invalid value. (move)'.format(players[ind]))

                            # send board
                            msg = '{} {}'.format(1, board2str(board))
                            if verbose:
                                print('[send : {}] : {}'.format(players[ind], msg))

                            rep = player_p[i].send(msg).strip()
                            if verbose:
                                print('[recv : {}] : {}'.format(players[ind], rep))
                            if rep != str(0):
                                raise ValueError(
                                    '{} reply invalid value. (board Ping)'.format(players[ind]))
    # end of initialize ###################################################################
                    if verbose:
                        print('Game Start!!')
                    while True:
                        for i, ind in enumerate(index):
                            # send board and get order
                            msg = '{} {}'.format(str(i+1), board2str(board))
                            if verbose:
                                print('[send : {}] : {}'.format(
                                    players[ind], msg))
                            order = player_p[i].send(msg).strip()
                            if verbose:
                                print('[recv : {}] : {}'.format(
                                    players[ind], order.strip()))
                            # judge invalid order
                            val = valid_str(board, i+1)
                            # check if there is no valid move
                            if len(val) == 0:
                                if order != ' '.join(['-1']*BOARD_SIZE):
                                    raise ValueError(
                                        'invalid move. have to be -1 -1 -1')
                            else:
                                if not(order in val):
                                    raise ValueError('invalid move')

                            order = list(map(int, order.strip().split(' ')))
                            # if not(all(np.array(order) >= 0) and all(np.array(order) < 3)):
                            #     raise ValueError('invalid order')
                            # forward 1 step
                            board = forward(board, order, i+1)
                            if verbose:
                                print('[board] {}'.format(board2str(board)))
                            f.write(' '.join(map(str, map(int, board)))+'\n')
                            # judge
                            res = judge(board, len(index))
                            if res:
                                break
                        if res:
                            s = calc_score(index, i+1)
                            if two_flag:
                                s = np.array([s[::2].sum(), s[1::2].sum()])
                            scores += s
                            break

                except (ValueError, TimeoutError):
                    if args.debug:
                        traceback.print_exc()
                        return 1
                    else:
                        s = calc_score(index, -(i+1),two_flag=two_flag)
                        if two_flag:
                            s = np.array([s[::2].sum(), s[1::2].sum()])
                        scores += s

        # board3d = board.reshape([3, 3, 3])

        if args.graphical:
            plt.cla()
            plt.bar(np.arange(len(scores))+1, scores,
                    tick_label=Players, align='center')
            plt.show()
        else:
            print(scores)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EOFError:
        sys.exit(0)
