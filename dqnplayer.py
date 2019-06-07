import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self,h,w,outputs):
        super(DQN,self).__init__()
        pass
    
    def forward(self,x):
        pass

