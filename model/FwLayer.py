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

class LinearFw(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super(LinearFw, self).__init__(in_features, out_features, bias= bias)

        self.weight.fast = None
        if bias:
            self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            if self.bias is not None and self.bias.fast is not None:
                out = F.linear(x, self.weight.fast, self.bias.fast)
            else:
                out = F.linear(x, self.weight.fast)
        else:
            out = super(LinearFw, self).forward(x)
        return out


class MLPFw(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, act_type = "relu"):
        super(MLPFw, self).__init__()
        self.fc1 = LinearFw(in_ch, hid_ch)
        self.act_layer = get_act_layer(act_type)
        self.fc2 = LinearFw(hid_ch, out_ch)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        return self.fc2(x)