
import torch
import torch.nn as nn
import numpy as np

class DoubletLoss(nn.Module):
    def __init__(self):
        super(DoubletLoss, self).__init__()
    def forward(self, x1, margin):
        distance_x1_0 =torch.sqrt(torch.sum(x1**2, axis=1))

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]