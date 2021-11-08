import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

class spatialGCN(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(spatialGCN, self).__init__()
        self.zetac = nn.Conv2d(in_channels, s_factor, 1)
        self.zetaN = nn.Conv2d(in_channels, s_factor, 1)
        self.conv = nn.Conv2d(s_factor, in_channels, 1)
        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels+s_factor, 1, padding='valid'),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        B, channel, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        C = channel // 2

        theta = self.zetac(x)
        theta = theta.reshape(-1, H * W, C)

        nu = self.zetaN(x)
        nu = nu.reshape(-1, H * W, C)
        nu1, nu2 = nu.size(1), nu.size(2)
        nu_tmp = nu.reshape(-1, nu1 * nu2)
        nu_tmp = torch.softmax(nu_tmp, -1)
        nu = nu_tmp.reshape(-1, nu1, nu2)

        xi = self.zetac(x)
        xi = xi.reshape(-1, H * W, C)
        xi1, xi2 = xi.size(1), xi.size(2)
        xi_tmp = xi.reshape(-1, xi1 * xi2)
        xi_tmp = torch.softmax(xi_tmp, -1)
        xi = xi_tmp.reshape(-1, xi1, xi2)
        xi = xi.permute(0, 2, 1)
        theta = theta.permute(0, 2, 1)

        F_s = torch.matmul(nu, xi)
        AF_s = torch.matmul(theta, F_s)
        AF_s = AF_s.reshape(-1, H, W, C)
        AF_s = AF_s.permute(0, 3, 1, 2)
        F_sGCN = self.conv(AF_s)
        return x + F_sGCN

class channelGCN(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(channelGCN, self).__init__()
        self.zetac = nn.Conv2d(in_channels, s_factor, 1)
        self.thtoth = nn.Conv2d(s_factor, s_factor, 1)
        self.zetaN = nn.Conv2d(in_channels, s_factor//2, 1)
        self.conv = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(18, 72, 1, 1)
        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels+s_factor, 1),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1))

    def forward(self, x):
        B, channel, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        C = channel // 2
        N = channel // 4
        input = x
        x = self.zetac(x)
        zeta = x.reshape(B, -1, C)

        x = self.zetaN(input)
        x = x.reshape(B, -1, N)
        kappa = x.permute(0, 2, 1)
        F_c = torch.matmul(kappa, zeta)

        F_c_tmp = F_c.reshape(-1, C*N)
        F_c_tmp = torch.softmax(F_c_tmp, -1)
        F_c = F_c_tmp.reshape(-1, N, C)

        F_c = F_c.unsqueeze(1)
        F_c = F_c + self.conv(F_c)
        F_c = torch.relu(F_c)
        F_c = F_c.permute(0, 3, 1, 2)

        F_c = self.thtoth(F_c)
        F_c = F_c.reshape(B, C, N)

        zeta = zeta.permute(0, 1, 2)
        F_c = torch.matmul(zeta, F_c)

        F_c = F_c.unsqueeze(1)
        F_c = F_c.reshape(B, -1, H, N)
        F_c = F_c.permute(0, 3, 2, 1)
        F_cGCN = self.conv2(F_c)
        return input + F_cGCN

class BasicUnit(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(BasicUnit, self).__init__()
        self.F_sGCN = spatialGCN(in_channels, s_factor)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, dilation=(1, 1), padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, dilation=(1, 1), padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, dilation=(3, 3), padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels, in_channels, 3, dilation=(3, 3), padding=3)
        self.relu4 = nn.ReLU()

        self.F_DCM = nn.Conv2d(in_channels, in_channels, 1, padding=1)
        self.reluF = nn.ReLU()
        self.cod = nn.Conv2d(in_channels * 5, in_channels, 3, dilation=(1, 1), padding=1)
        self.F_cGCN = channelGCN(in_channels, s_factor)

    def forward(self, x):

        F_sGCN = self.F_sGCN(x)
        # F_sGCN = self.orsnet(F_sGCN)
        con1 = self.conv1(F_sGCN)
        reu1 = self.relu1(con1)

        con2 = self.conv2(reu1)
        reu2 = self.relu2(con2)

        con3 = self.conv3(F_sGCN)
        reu3 = self.relu3(con3)

        con4 = self.conv4(reu3)
        reu4 = self.relu4(con4)

        F_sGCN = F_sGCN.permute(0, 2, 3, 1)
        reu1 = reu1.permute(0, 2, 3, 1)
        reu2 = reu2.permute(0, 2, 3, 1)
        reu3 = reu3.permute(0, 2, 3, 1)
        reu4 = reu4.permute(0, 2, 3, 1)
        tmp = torch.cat((F_sGCN, reu1, reu2, reu3, reu4), -1)
        tmp = tmp.permute(0, 3, 1, 2)
        F_DCM = self.cod(tmp)
        F_DCM = self.relu1(F_DCM)
        F_cGCN = self.F_cGCN(F_DCM)
        F_unit = F_cGCN + x
        return F_unit

class DualGCN(nn.Module):
    
    def __init__(self, recurrent_iter = 6, use_gpu = True, num_channels = 32):

        super(DualGCN, self).__init__()
        
        self.iteration = recurrent_iter
        self.use_gpu = use_gpu
        self.ncs = num_channels

        self.conv_0 = nn.Sequential(
            nn.Conv2d(1, 72, 3, 1, 1),
            )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(72, 72, 3, 1, 1),
            )
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(144, 72, 1),
            )
        self.BU = BasicUnit(72, 36)
        self.cond = nn.Conv2d(144, 72, 1, 1)
        self.end = nn.Conv2d(72, 1, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = inputs
        x_list = []
        fea0 = self.conv_0(x)
        x = fea0
        x1 = self.conv_1(x)
        e0 = self.BU(x1)
        e1 = self.BU(e0)
        e2 = self.BU(e1)
        e3 = self.BU(e2)
        e4 = self.BU(e3)
        m = self.BU(e4)
        d4 = torch.cat((m, e4), 1)
        d4 = self.conv_d4(d4)
        d4 = self.BU(d4)

        d3 = torch.cat((d4, e3), 1)
        d3 = self.conv_d4(d3)
        d3 = self.BU(d3)

        d2 = torch.cat((d3, e2), 1)
        d2 = self.conv_d4(d2)
        d2 = self.BU(d2)

        d1 = torch.cat((d2, e1), 1)
        d1 = self.conv_d4(d1)
        d1 = self.BU(d1)

        d0 = torch.cat((d1, e0), 1)
        d0 = self.conv_d4(d0)
        d0 = self.BU(d0)

        de = torch.cat((d0, x1), 1)
        de = self.cond(de)
        de = self.relu(de)
        de = de + fea0
        de = self.end(de)

        output = inputs + de

        x_list.append(output)
        return output, x_list