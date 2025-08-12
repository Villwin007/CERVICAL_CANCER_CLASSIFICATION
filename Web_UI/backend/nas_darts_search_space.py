from nas_operations import OPS
import torch.nn as nn
import torch
import torch.nn.functional as F

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride, True)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
