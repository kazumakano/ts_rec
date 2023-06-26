import torch
from torch import nn


class FullNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(3 * 22 * 17, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.layer1(input.view(-1, 3 * 22 * 17))

        return output
