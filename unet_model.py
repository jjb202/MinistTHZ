""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""


import torch
import torch.nn as nn



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=1,norm_layer = nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False
                              )  # verify bias false
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class BasicTransConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=1,norm_layer = nn.BatchNorm2d):
        super(BasicTransConv2d, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_planes, out_planes,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, bias=False)
        self.bn = norm_layer(out_planes)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.bn(self.transconv(x)))
        return x

class UnetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=16,
                 norm_layer=nn.BatchNorm2d):
        super(UnetGenerator, self).__init__()
        self.init_conv = BasicConv2d(input_nc, ngf, kernel_size= 3 , stride = 1, padding=1,norm_layer = norm_layer)# [h,w]
        self.down1 = nn.Sequential(
            BasicConv2d(ngf, ngf, kernel_size= 3 , stride = 1, padding=1,norm_layer = norm_layer),
            nn.MaxPool2d(2)
        ) # [h/2,w/2]=14*14
        self.down2 = nn.Sequential(
            BasicConv2d(ngf, ngf*2, kernel_size= 3 , stride = 1, padding=1,norm_layer = norm_layer),
            nn.MaxPool2d(2)
        )  # [h/4,w/4]=7*7
        self.keep = BasicConv2d(ngf*2, ngf*2, kernel_size= 3 , stride = 1, padding=1,norm_layer = norm_layer)# [h/4,w/4]=7*7

        self.up2 = BasicTransConv2d(ngf*4,ngf, kernel_size=4, stride=2, padding=1, norm_layer = norm_layer)# [h/2,w/2]

        self.up1 = BasicTransConv2d(ngf*2,ngf, kernel_size=4, stride=2, padding=1, norm_layer = norm_layer)# [h,w]

        self.out_conv = nn.Conv2d(ngf, output_nc, 3, 1, 1)

    def forward(self, x):
        o1 = self.init_conv(x) # [h,w]
        o2 =  self.down1(o1)  # [h/2,w/2]
        o3 = self.down2(o2) # [h/4,w/4]
        o4 =  self.keep(o3) # [h/4,w/4]
        in1 = torch.cat((o3, o4), 1)
        up1 =  self.up2(in1) # [h/2,w/2]
        in2 = torch.cat((up1, o2), 1)
        up2 = self.up1(in2)# [h,w]
        out = self.out_conv(up2)# [h,w]
        return out
if __name__ == '__main__':
    # net = UNet()
    # print(net)
    net = UnetGenerator(3,3)
    x = torch.rand(1, 3, 28, 28)
    y = net(x)
    print(y.shape)
