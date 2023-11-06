"""普通のUNet。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import os
import csv
import pandas as pd
import soundfile

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class UNetNormal(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        # Deconv
        self.deconv6_a = nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv6_b = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv5_a = nn.ConvTranspose2d(512, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_b = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv4_a = nn.ConvTranspose2d(256, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_b = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv3_a = nn.ConvTranspose2d(128, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_b = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.deconv2_a = nn.ConvTranspose2d(64, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_b = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size = (5, 5), stride=(2, 2), padding=2)
        #deviceを指定
        self.to(device)
        
        
    def forward(self, input):
        # Encoder
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        # Decoder
        deconv6_out = self.deconv6_a(conv6_out, output_size = conv5_out.size())
        deconv6_out = self.deconv6_b(deconv6_out)
        deconv5_out = self.deconv5_a(torch.cat([deconv6_out, conv5_out], 1), output_size = conv4_out.size())
        deconv5_out = self.deconv5_b(deconv5_out)
        deconv4_out = self.deconv4_a(torch.cat([deconv5_out, conv4_out], 1), output_size = conv3_out.size())
        deconv4_out = self.deconv4_b(deconv4_out)
        deconv3_out = self.deconv3_a(torch.cat([deconv4_out, conv3_out], 1), output_size = conv2_out.size())
        deconv3_out = self.deconv3_b(deconv3_out)
        deconv2_out = self.deconv2_a(torch.cat([deconv3_out, conv2_out], 1), output_size = conv1_out.size())
        deconv2_out = self.deconv2_b(deconv2_out)
        deconv1_out = self.deconv1(torch.cat([deconv2_out, conv1_out], 1), output_size = input.size())
        output = torch.sigmoid(deconv1_out)
        return output
    
        