import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
import game
from tensorboardX import SummaryWriter
import argparse
import os
from tqdm import tqdm
import random
from miwa.abplayer import abplayer
import sys

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

class FCN(nn.Module):
    def __init__(self, once=False):
        super(FCN, self).__init__()
        # save predictions
        self.preds = []
        self.masks = []
        self.once = once

        # at first, each face parse
        # weight is shared
        # input 3x3x3
        self.conv1 = nn.Conv3d(1, 8, kernel_size=2, stride=1, padding=1)
        # 4x4x4
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 32, kernel_size=2, stride=1, padding=1)
        # 5x5x5
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=2, stride=1, padding=1)
        # 6x6x6
        self.bn3 = nn.BatchNorm3d(32)

        # 6x6x6
        self.conv3d1 = nn.Conv3d(32 * 4, 64, kernel_size=2, stride=1)
        # 5x5x5
        self.bn3d1 = nn.BatchNorm3d(64)
        self.conv3d2 = nn.Conv3d(64, 32, kernel_size=2, stride=1)
        # 4x4x4
        self.bn3d2 = nn.BatchNorm3d(32)
        self.conv3d3 = nn.Conv3d(32, 1, kernel_size=2, stride=1)
        # 3x3x3
        self.drop = nn.Dropout3d(0.2)

    def forward(self, x, mask):
        # TODO: concat order
        # input: batch x player x h x w x d
        num_player = x.size(1)
        x = [x[:, i].unsqueeze(1) for i in range(num_player)]
        for i in range(num_player):
            x[i] = F.relu(self.bn1(self.conv1(x[i])))
            x[i] = F.relu(self.bn2(self.conv2(x[i])))
            x[i] = F.relu(self.bn3(self.conv3(x[i])))

        # batch x player x h x w x d x channel
        x = torch.cat([x[i] for i in range(num_player)], dim=1)
        # batch x h x w x d x (player * channel)
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.drop(x)
        x = self.conv3d3(x)
        x = x.squeeze(dim=1)

        # softmax
        x_ = x.exp()
        x_sum = x_.sum(dim=-1).sum(dim=-1).sum(dim=-1)
        for i in range(x.size(0)):
            x[i] = torch.div(x_[i], x_sum[i])

        # save predictions and mask
        if self.training and self.once:
            self.preds.append(x.clone())
            self.masks.append(mask)

        x = x * mask
        return x

    def get_last_pred(self):
        if preds:
            return None
        else:
            return self.preds[-1]

    def get_preds(self):
        return self.preds

    def get_masks(self):
        return self.masks

    def reset(self):
        self.masks = []
        self.preds = []

    def set_preds(self, preds):
        self.preds = preds

    def set_masks(self, masks):
        self.masks = masks


def make_regal_mask(board, num):
    moves = game.valid_move(board, num)
    mask = torch.zeros(3, 3, 3)
    for move in moves:
        mask[move[0], move[1], move[2]] = 1.
    return mask


def line2index(num):
    return [num // 9, num % 9 // 3, num % 3]


# ind is like (my_num, next_player, next, next)
def feature(board, ind):
    b = np.array(board)
    f = [(b == i).astype(float).reshape(3, 3, 3) for i in ind]
    return np.array(f)


model = FCN(once=True).to(device)
opt = optim.Adam(model.parameters())


def mlplayer_once(msg, epsilon=0.95):
    board = np.array(msg[1:])
    idx = msg[0]
    if not model.training:
        epsilon = 1.
    if torch.bernoulli(torch.tensor(epsilon)).item():
        index = [(i + idx - 1) % 4 for i in range(4)]
        x = feature(board, ind=index)
        x = torch.tensor(x).to(torch.float).to(device)
        x = x.unsqueeze(0)
        mask = make_regal_mask(board, idx).to(device)
        if mask.sum() == 0:
            return [-1, -1, -1]
        mask = mask.unsqueeze(0)
        y = model(x, mask)
        y = y.squeeze()
        # print(y)

        best_move = line2index(y.argmax().item())
    else:
        val = game.valid_move(board, idx)
        if not val:
            best_move = [-1, -1, -1]
        else:
            best_move = random.choice(val)
    return best_move


def mlplayer_unit(board, num):
    index = [(i + num - 1) % 4 for i in range(4)]
    x = feature(board, ind=index)
    x = torch.tensor(x).to(torch.float).to(device)
    x = x.unsqueeze(0)
    mask = make_regal_mask(board, idx).to(device)
    if mask.sum() == 0:
        return [-1, -1, -1]
    mask = mask.unsqueeze(0)
    y = model(x, mask)
    y = y.squeeze()

    return y


def mlplayer_search(msg):
    idx = msg[0]
    board = np.array(msg[1:])
    index = [(i + idx - 1) % 4 for i in range(4)]
    x = feature(board, ind=index)
    x = torch.tensor(x).to(torch.float).to(device)
    x = x.unsqueeze(0)
    mask = make_regal_mask(board, idx).to(device)
    if mask.sum() == 0:
        return [-1, -1, -1]
    mask = mask.unsqueeze(0)
    y = model(x, mask)
    y = y.squeeze()

    best_move = line2index(y.argmax().item())
    return best_move


def move_hand(board, hand, num):
    hand3d = line2index(hand)
    board[hand3d[0], hand3d[1], hand3d[2]] = num
    return board


def beam_search(boards, num, depth, width):
    b = []
    for board in boards:
        if not (all(board != 0)):
            b.append(boards)
    boards = b
    if depth == 0 or (not boards):
        return 0, None
    y = mlplayer_unit(boards, num)
    v_size = list([*y.size()[:-3].to_list(), 3**3])
    y = y.view(v_size)
    hands = y.topk(width)[-1]
    for board, hand in zip(boards, hands):
        new_boards = torch.tensor([move_hand(board, h, num)
                                   for h in hand]).to(board)
        beam_search(new_boards, num % 4 + 1, depth - 1, width)
    return 0


is_miwa = True
miwa_dp = 4


def train_once(opp):
    model.train()
    opt.zero_grad()

    if is_miwa:
        miwa = abplayer('2 1 3', miwa_dp)
        opp = miwa.play
    # first game
    res1 = game.game(opp, mlplayer_once)
    masks1 = model.get_masks()
    preds1 = model.get_preds()
    model.reset()

    loss1 = 0
    loss2 = 0
    if res1 == 0:
        # if win
        for i, (p, m) in enumerate(zip(preds1, masks1)):
            p = p.squeeze()
            m = m.squeeze()
            loss1 = (1 - p[line2index(p.argmax())])**2
            # loss1 += p[m == 1].mean()
    elif res1 == 1:
        # if lose
        for i, (p, m) in enumerate(zip(preds1, masks1)):
            p = p.squeeze()
            m = m.squeeze()
            loss1 = (p[line2index(p.argmax())] - 1)**2
            # loss1 += p[m == 1].mean()
    else:
        # if draw
        for i, (p, m) in enumerate(zip(preds1, masks1)):
            p = p.squeeze()
            m = m.squeeze()
            loss1 = p[m == 1].mean()

    # second game
    res2 = game.game(mlplayer_once, opp)
    masks2 = model.get_masks()
    preds2 = model.get_preds()
    model.reset()

    if is_miwa:
        miwa = abplayer('2 2 4', miwa_dp)
        opp = miwa.play

    if res2 == 1:
        # if lose
        for i, (p, m) in enumerate(zip(preds2, masks2)):
            p = p.squeeze()
            m = m.squeeze()
            loss2 = (1 - p[line2index(p.argmax())])**2
            # loss2 += p[m == 1].mean()
    elif res2 == 0:
        # if win
        for i, (p, m) in enumerate(zip(preds2, masks2)):
            p = p.squeeze()
            m = m.squeeze()
            loss2 = (p[line2index(p.argmax())] - 1)**2
            # loss2 += p[m == 1].mean()
    else:
        # if draw
        for i, (p, m) in enumerate(zip(preds2, masks2)):
            p = p.squeeze()
            m = m.squeeze()
            loss2 = p[m == 1].mean()

    # training
    loss = loss1 + loss2

    return loss


def eval_once(opp, times=1):
    model.eval()
    win = 0
    lose = 0
    draw = 0
    for t in range(times):
        if is_miwa:
            miwa = abplayer('2 1 3', miwa_dp)
            opp = miwa.play
        res1 = game.game(opp, mlplayer_once)
        if res1 == 0:
            win += 1
        elif res1 == 1:
            lose += 1
        else:
            draw += 1

        if is_miwa:
            miwa = abplayer('2 2 4', miwa_dp)
            opp = miwa.play
        res2 = game.game(mlplayer_once, opp)
        if res2 == 1:
            win += 1
        elif res2 == 0:
            lose += 1
        else:
            draw += 1
    return win, lose, draw


def cal_l2loss(parameters):
    l2loss = 0
    for p in parameters:
        l2loss += p.norm()
    return l2loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        required=False,
                        choices=['train', 'test'],
                        default='test')
    parser.add_argument('--epoch', type=int, required=False, default=1)
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--tb_path', type=str, required=False, default='runs/')
    parser.add_argument('--out_path',
                        type=str,
                        required=False,
                        default='model/')
    parser.add_argument('--epsilon', type=float, required=False, default=0.)
    parser.add_argument('--alpha', type=float, required=False, default=0.01)
    parser.add_argument('--weight',
                        type=str,
                        required=False,
                        default='makino/pretrained_model')
    args = parser.parse_args()

    mode = args.mode
    epochs = args.epoch
    batch_size = args.batch_size
    tb_path = args.tb_path
    out_path = args.out_path
    alpha = args.alpha
    weight_path = args.weight
    epsilon = args.epsilon
    if mode == 'train':
        dirs = os.listdir(tb_path)

        if dirs == []:
            num = 1
        else:
            num = max(map(int, dirs)) + 1

        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
        else:
            print('WARNING: mlplayer weight is not exist.', file=sys.stderr)

        tb_path = os.path.join(tb_path, str(num).zfill(4))
        out_path = os.path.join(out_path, str(num).zfill(4))
        os.makedirs(out_path)
        os.makedirs(tb_path)

        writer = SummaryWriter(tb_path)

        for epoch in range(epochs):
            loss = 0
            l2loss = 0
            for b in tqdm(range(batch_size)):
                loss += train_once(game.random_player)
                l2loss += alpha * cal_l2loss(model.parameters())
            loss += l2loss
            loss /= batch_size
            loss.backward()
            opt.step()
            writer.add_scalar('train/loss', loss, epoch)
            win, lose, draw = eval_once(game.random_player, times=10)
            writer.add_scalar('eval/win', win, epoch)
            writer.add_scalar('eval/lose', lose, epoch)
            writer.add_scalar('eval/draw', draw, epoch)
            writer.add_scalar('eval/rate', win / (win + lose), epoch)
            torch.save(model.state_dict(),
                       os.path.join(out_path,
                                    str(epoch).zfill(3)))
    if mode == 'test':
        model.eval()
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
        else:
            print('WARNING: mlplayer weight is not exist.', file=sys.stderr)
        get = list(map(int, input().strip().split(' ')))
        player_num = get[0]
        my_turn = get[1:]
        print(' '.join(map(str, my_turn)))

        board = np.array(list(map(int, input().strip().split(' '))))
        piece = np.zeros(3)

        print(0)
        try:
            while True:
                msg = np.array(list(map(int, input().strip().split(' '))))

                move = mlplayer_once(msg)
                print(' '.join(map(str, move)))
        except EOFError:
            pass
