import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # Inspired from the Alexnet implementation in pytorch. Somewhat reduced.
    def __init__(self, classes = 2, dropout = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=11, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.LazyConv2d(192, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.LazyLinear(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.LazyLinear(4096),
            nn.ReLU(inplace=True),
            nn.LazyLinear(classes)
        )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.classifier(x)
            return x
    


