import torch
import torch.nn as nn
import torch.nn.functional as F

from nas_darts_search_space import OPS

class Cell(nn.Module):
    def __init__(self, C, stride, affine=True):
        super(Cell, self).__init__()
        self._ops = nn.ModuleList([
            op(C, stride, affine) for name, op in OPS.items()
        ])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class DARTSNetwork(nn.Module):
    def __init__(self, C_in=3, num_classes=5, num_cells=3, init_channels=16):
        super(DARTSNetwork, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, init_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels)
        )

        self.cells = nn.ModuleList()
        C_prev = init_channels
        for i in range(num_cells):
            stride = 1  # or 2 if you want spatial reduction
            cell = Cell(C_prev, stride=stride, affine=True)
            self.cells.append(cell)

        # Architecture parameters (alphas)
        self.alphas = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(len(OPS)))
            for _ in range(num_cells)
        ])

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(init_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)

        weights = [F.softmax(alpha, dim=0) for alpha in self.alphas]

        for i, cell in enumerate(self.cells):
            x = cell(x, weights[i])

        x = self.global_pooling(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
