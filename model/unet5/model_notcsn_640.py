"""640次元を条件付け、Decode。Decoderは1つ。csn使ってないから動かない。"""

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

from ..csn import ConditionalSimNet2d

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

class DeconvOther(nn.Module):
    def __init__(self, out_channels) -> None:
        super().__init__()
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
        )
    def forward(self, input):
        return self.deconv(input)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True),
        )
    def forward(self, input):
        return self.conv(input)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = Conv2d(1, 16)
        self.conv2 = Conv2d(16, 32)
        self.conv3 = Conv2d(32, 64)
        self.conv4 = Conv2d(64, 128)
        self.conv5 = Conv2d(128, 256)
        self.conv6 = Conv2d(256, 640)
        # 条件付け
        csn = ConditionalSimNet2d(mask, 5)
        # Decoder
        self.deconv6_a = nn.ConvTranspose2d(640, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv6_b = DeconvOther(256)
        self.deconv5_a = nn.ConvTranspose2d(256+256, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_b = DeconvOther(128)
        self.deconv4_a = nn.ConvTranspose2d(128+128, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_b = DeconvOther(64)
        self.deconv3_a = nn.ConvTranspose2d(64+64, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_b = DeconvOther(32)
        self.deconv2_a = nn.ConvTranspose2d(32+32, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_b = DeconvOther(16)
        self.deconv1_a = nn.ConvTranspose2d(16+16, 1, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv1 = DeconvOther(1)
        #deviceを指定
        self.to(device)
        #print(next(self.decoder["guitar"].parameters()).is_cuda)

    def forward(self, input):
        # Encoder
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        # Decoder
        # 特徴量を条件づけ
        size = conv6_out.shape
        mask_init = torch.ones(size).to(device)
        mask = torch.tensor([torch.cat([mask_init[:, :128, :, :],
                            torch.zeros(size[0], 512, size[2], size[3], device=device)], 1),

                torch.cat([torch.cat([torch.zeros(size[0], 128, size[2], size[3], device=device),
                                        mask_init[:, 128: 256, :, :]], 1),
                            torch.zeros(size[0], 384, size[2], size[3], device=device)], 1),

                torch.cat([torch.cat([torch.zeros(size[0], 256, size[2], size[3], device=device),
                                        mask_init[:, 256: 384, :, :]], 1),
                            torch.zeros(size[0], 256, size[2], size[3], device=device)], 1),

                torch.cat([torch.cat([torch.zeros(size[0], 384, size[2], size[3], device=device),
                                        mask_init[:, 384: 512, :, :]], 1),
                            torch.zeros(size[0], 128, size[2], size[3], device=device)], 1),

                torch.cat([torch.zeros(size[0], 512, size[2], size[3], device=device),
                            mask_init[:, 512: 640, :, :]], 1)
        ], device=device)
        
        inst_list = ["guitar", "drum", "base", "piano", "others"]
        output = {
            "guitar": None,
            "drum": None,
            "base": None,
            "piano": None,
            "others": None,
        }
        for inst in inst_list:
            sep_feature = csn()
            deconv6_out = self.deconv6_a(sep_feature[inst], output_size = conv5_out.size())
            deconv6_out = self.deconv6_b(deconv6_out)
            deconv5_out = self.deconv5_a(torch.cat([deconv6_out, conv5_out], 1), output_size = conv4_out.size())
            deconv5_out = self.deconv5_b(deconv5_out)
            deconv4_out = self.deconv4_a(torch.cat([deconv5_out, conv4_out], 1), output_size = conv3_out.size())
            deconv4_out = self.deconv4_b(deconv4_out)
            deconv3_out = self.deconv3_a(torch.cat([deconv4_out, conv3_out], 1), output_size = conv2_out.size())
            deconv3_out = self.deconv3_b(deconv3_out)
            deconv2_out = self.deconv2_a(torch.cat([deconv3_out, conv2_out], 1), output_size = conv1_out.size())
            deconv2_out = self.deconv2_b(deconv2_out)
            deconv1_out = self.deconv1_a(torch.cat([deconv2_out, conv1_out], 1), output_size = input.size())
            output[inst] = torch.sigmoid(deconv1_out)
        return output
    
    """
    def save(self, path):
        # Record the parameters
        print("Save the trained model to {}".format(path))
        state_encoder = {
            'conv1': self.conv1.state_dict(),
            'conv2': self.conv2.state_dict(),
            'conv3': self.conv3.state_dict(),
            'conv4': self.conv4.state_dict(),
            'conv5': self.conv5.state_dict(),
            'conv6': self.conv6.state_dict(),
            'deconv1': self.deconv1.state_dict(),
            'deconv2': self.deconv2.state_dict(),
            'deconv3': self.deconv3.state_dict(),
            'deconv4': self.deconv4.state_dict(),
            'deconv5': self.deconv5.state_dict(),
            'deconv6': self.deconv6.state_dict(),
        }

        # Record the optimizer and loss
        state['optim'] = self.optim.state_dict()
        for key in self.__dict__:
            if len(key) > 10:
                if key[1:9] == 'oss_list':
                    state[key] = getattr(self, key)
        torch.save(state, path)
        print("Finish saving model!")

    def load(self, path):
        if os.path.exists(path):
            print("Load the pre-trained model from {}".format(path))
            bar = progress_bar("Load model", 13)
            state = torch.load(path)
            for (key, obj) in state.items():
                if len(key) > 10:
                    if key[1:9] == 'oss_list':
                        setattr(self, key, obj)
            self.conv1.load_state_dict(state['conv1'])
            bar.update(1)
            self.conv2.load_state_dict(state['conv2'])
            bar.update(1)
            self.conv3.load_state_dict(state['conv3'])
            bar.update(1)
            self.conv4.load_state_dict(state['conv4'])
            bar.update(1)
            self.conv5.load_state_dict(state['conv5'])
            bar.update(1)
            self.conv6.load_state_dict(state['conv6'])
            bar.update(1)
            self.deconv1.load_state_dict(state['deconv1'])
            bar.update(1)
            self.deconv2.load_state_dict(state['deconv2'])
            bar.update(1)
            self.deconv3.load_state_dict(state['deconv3'])
            bar.update(1)
            self.deconv4.load_state_dict(state['deconv4'])
            bar.update(1)
            self.deconv5.load_state_dict(state['deconv5'])
            bar.update(1)
            self.deconv6.load_state_dict(state['deconv6'])
            bar.update(1)
            self.optim.load_state_dict(state['optim'])
            bar.update(1)
            print("Finish loading model!")
        else:
            raise MyError("Pre-trained model {} is not exist...".format(path))
    """
if "__main__" == __name__:
    # モデルを定義
    model = UNet()
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 128, 8196*3),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)