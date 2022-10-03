import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform, zeros
from .FwLayer import get_act_layer


class FC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FC, self).__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, act_type = "relu"):
        super(MLP, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        self.act_layer = get_act_layer(act_type)
        self.fc2 = FC(hid_ch, out_ch)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        return self.fc2(x)

