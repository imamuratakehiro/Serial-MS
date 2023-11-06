"""2次元特徴量をグローバル平均プーリングで1次元化"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import os
#import stempeg
import csv
import pandas as pd
#import soundfile

class AVGPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, input):
        # インスタンスを生成していなかったら、生成する。
        if not "argpool" in vars(self):
            self.avgpool = nn.AvgPool2d(kernel_size=(input.shape[2], input.shape[3]))
        # グローバル平均プーリング
        output = self.avgpool(input)
        return output[:,:,0,0]