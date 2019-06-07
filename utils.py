
def get_piece(board3d, player):
    b = (board3d == player).astype(int)
    piece = b.sum((0, 1))
    return piece

def board23d(board):
    board3d = board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
    return board3d

def valid_move(board, player):
    board3d = board23d(board)
    piece = get_piece(board3d)
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


def forward(board, order, player_idx):
    if order[0] == -1:
        return board
    board3d = board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
    if board3d[order[0], order[1], order[2]] == 0:
        board3d[order[0], order[1], order[2]] = player_idx
    else:
        assert 'can not forward board, fix valid_move'

    return board
