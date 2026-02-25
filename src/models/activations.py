import torch
import torch.nn as nn
from math import sqrt

class Heo:
    class HeLU2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            self.alpha = nn.Parameter(torch.full((c,), 1.0))
            self.beta = nn.Parameter(torch.full((c,), -1.0))
            self.redweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.75))
            self.blueweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.75))

            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raw = x
            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            red = (torch.tanh(sqrt(3.0) * self.redweight) + 1.0).view(1, -1, 1, 1)
            blue = (torch.tanh(sqrt(3.0) * self.blueweight) + 1.0).view(1, -1, 1, 1)
            x = rgx * red + bgx * blue

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            y = (alpha * x + beta * raw) / 2
            
            return y