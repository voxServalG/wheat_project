import torch
import torch.nn as nn
import torch.nn.functional as F

# (3,256,256) -> 1

def Hswish(x, inplace=True):
    return x * F.relu6(x+3., inplace=inplace) /6.

def Hsigmoid(x, inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.

#squeeze and excite
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4) -> None:
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.se(y)
        y = Hsigmoid(y).view(b,c,1,1)
        return x * y.expand_as(x)
    
class Bottleneck(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)