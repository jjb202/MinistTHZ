#PyTorch lib
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
from blocks import DFM
from blocks import _NonLocalBlockND
from blocks import ECAAttention
from blocks import SAM
from resnet_cbam import BasicBlock

class PMANN(nn.Module):
    def __init__(self, recurrent_iter=6):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.att = _NonLocalBlockND(32, inter_channels=None, dimension=2)
        # self.att2 = BasicBlock(64, 64)
        # self.sam = SAM(1, kernel_size=1, bias=False)
        act = nn.PReLU()
        self.orsnet = DFM(32, 32, 28, 4, act, False, 20, 12)
        self.conv0 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),

            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            act,
            nn.Conv2d(32, 32, 3, 1, 1),
            act
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = torch.zeros(batch_size, 32, row, col)# RNN 中使用，保存输出值和状态
        c = torch.zeros(batch_size, 32, row, col)

        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()

        x_list = []

        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            x = torch.cat((x, h), 1)
            x = self.orsnet(x)
            # x = self.att2(x)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            x = self.att(x)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)
            # ## Apply SAM
            # x, stage2_img = self.sam(x, input)
            x = x + input
            x_list.append(x)

        return x, x_list
