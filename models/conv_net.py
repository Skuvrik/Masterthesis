import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ConvNet, self).__init__()
        self.in_planes = in_planes
        self.dropRate = dropRate

    


