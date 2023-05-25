import torch
import torch.nn as nn


class TISRoverPlus(nn.Module):
    def __init__(self):
        super(TISRoverPlus, self).__init__()
        self.features = \
            nn.Sequential(\
                # 1
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 4)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 1)),
                # 2
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 1)),
                # 3
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 1))
                )

        self.classifier = \
            nn.Sequential(
                nn.Linear(in_features=768, out_features=128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=128, out_features=2))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
