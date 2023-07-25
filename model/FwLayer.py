import torch.nn as nn
import torch.nn.functional as F


def get_act_layer(act_type: str):
    if act_type == "relu":
        return nn.ReLU()
    elif act_type == "tanh":
        return nn.Tanh()
    elif act_type == "leaky_relu":
        return nn.LeakyReLU()
    elif act_type == "prelu":
        return nn.PReLU()
    elif act_type == 'grelu':
        return nn.GELU()
    elif act_type == "none":
        return lambda x : x
    else:
        raise NotImplementedError("Error: %s activation function is not supported now." % (act_type))


