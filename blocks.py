import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn import Parameter
from cupy_layers.aggregation_zeropad import LocalConvolution

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)

class MA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.num_heads * self.coef
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x)  # B, N, C
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False, kernel_size=3):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
        self.key_embed = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=3//2, groups=8, bias=False),
            nn.ReLU(inplace=True)
        )
        self.kernel_size = kernel_size
        self.dw_group = 2
        share_planes = 8
        factor = 2
        dim = 64
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, groups=self.dw_group, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(3, 2) * dim // share_planes, kernel_size=1, groups=self.dw_group),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(3, 2) * dim // share_planes)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.dw_group, bias=False)
        )
        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        k = self.key_embed(x)
        qk = torch.cat([x.unsqueeze(2), k.unsqueeze(2)], dim=2)
        qk = qk.view(batch_size, -1, height, width)

        w = self.embed(qk)
        w = w.view(batch_size * self.dw_group, 1, -1, self.kernel_size * self.kernel_size, height, width)

        x = self.conv1x1(x)
        x = x.view(batch_size * self.dw_group, -1, height, width)
        x = self.local_conv(x, w)
        x = x.view(batch_size, -1, height, width)

        # B, C, H, W = x.shape
        # p = x.view(B, C, 1, H, W)
        # k = k.view(B, C, 1, H, W)
        # p = torch.cat([p, k], dim=2)
        #
        # x_gap = p.sum(dim=2)
        # x_gap = x_gap.mean((2, 3), keepdim=True)

        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab, dim=64):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(dim, 3, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(dim, dim, 3))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class DFM(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab, dim=64):
        super(DFM, self).__init__()

        self.orb1 = ORB(dim, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(dim, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(dim, 0)
        self.up_dec1 = UpSample(dim, 0)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(dim, dim,  kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(dim, dim,  kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(dim, dim,  kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.orb1(x)
        x = x + self.conv_enc1(x) + self.conv_dec1(x)
        x = self.orb2(x)
        x = x + self.conv_enc2(x) + self.conv_dec2(x)
        x = self.orb3(x)
        x = x + self.conv_enc3(x) + self.conv_dec3(x)
        x = self.orb3(x)
        x = x + self.conv_enc3(x) + self.conv_dec3(x)
        x = self.orb3(x)
        x = x + self.conv_enc3(x) + self.conv_dec3(x)
        return x


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer,)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer,)


class convBlockVGG_ND(nn.Module):

    def __init__(self, 
        num_channels = [32, 64], 
        is_batchnorm = True,
        dimension = 3,
        kernel_size = 3,
        stride = 1,
        padding = 1
    ):

        super(convBlockVGG_ND, self).__init__()

        self.num_channels = num_channels
        self.is_batchnorm = is_batchnorm
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if self.dimension == 1:
            conv_ND = nn.Conv1d
            batchNorm_ND = nn.BatchNorm1d
        elif self.dimension == 2:
            conv_ND = nn.Conv2d
            batchNorm_ND = nn.BatchNorm2d
        elif self.dimension == 3:
            conv_ND = nn.Conv3d
            batchNorm_ND = nn.BatchNorm3d

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                conv_ND(self.num_channels[0], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                batchNorm_ND(self.num_channels[1]), 
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                conv_ND(self.num_channels[1], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                batchNorm_ND(self.num_channels[1]), 
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                conv_ND(self.num_channels[0], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                conv_ND(self.num_channels[1], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                nn.ReLU(inplace=True)
            )

    def forward(self, inputs):

        outputs = self.conv2(self.conv1(inputs))

        return outputs

class unetConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, is_batchnorm, conv_type = 'vgg'):
        
        super(unetConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm
        self.conv_type = conv_type
        self.convolution = None

        if self.conv_type == 'vgg':
            self.convolution = convBlockVGG_ND(
                num_channels = [self.in_channels, self.out_channels],
                is_batchnorm = self.is_batchnorm,
                dimension = 2
            )

    def forward(self, inputs):

        outputs = self.convolution(inputs)

        return outputs
        
class unetUp2d(nn.Module):

    def __init__(self, in_size, out_size, is_deconv, is_hpool = True):
        
        super(unetUp2d, self).__init__()
        
        self.conv = unetConv2d(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size = 2, stride = 2, padding = 0)
        else:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def forward(self, leftIn, rightIn):
        
        rightIn = self.up(rightIn)
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)
        
        pad = torch.nn.ConstantPad2d(padding, 0)
        rightIn = pad(rightIn)

        lrCat = torch.cat([leftIn, rightIn], 1)
        output = self.conv(lrCat)
        
        return output 

class fullEleAtt(nn.Module):

    def __init__(self, in_channels, is_conv = False):  

        super(fullEleAtt, self).__init__()

        self.in_cs = in_channels
        self.is_conv = is_conv

        if self.is_conv:

            self.theta_conv = nn.Conv2d(in_channels = self.in_cs, out_channels = self.in_cs, kernel_size = 1, stride = 1, padding = 0) 
            self.phi_conv = nn.Conv2d(in_channels = self.in_cs, out_channels = self.in_cs, kernel_size = 1, stride = 1, padding = 0) 
            self.g_conv = nn.Conv2d(in_channels = self.in_cs, out_channels = self.in_cs, kernel_size = 1, stride = 1, padding = 0) 

        self.vector_width   = vectorAttBlockWidth()
        self.vector_height  = vectorAttBlockHeight()
        self.vector_channel = vectorAttBlockChannel()

        self.vector_atts = [self.vector_width, self.vector_height, self.vector_channel]

        self.beta_width   = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        self.beta_height  = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        self.beta_channel = nn.Parameter(torch.tensor(0.0, requires_grad = True))

        self.betas = [self.beta_width, self.beta_height, self.beta_channel]

    def forward(self, x):

        for idx, vector_att in enumerate(self.vector_atts):

            if self.is_conv:
                theta_x = self.theta_conv(x)
                phi_x = self.phi_conv(x)
                g_x = self.g_conv(x)

                x_t = vector_att(theta_x, phi_x, g_x)
                x = x + x_t * self.betas[idx]

            else:
                
                x_t = vector_att(x, x, x)
                x = x + x_t * self.betas[idx]

        return x

class vectorAttBlockWidth(nn.Module):

    def __init__(self):

        super(vectorAttBlockWidth, self).__init__()

    def forward(self, x_, x_t, g_x):

        # n * c * h * w
        batch_size = x_.size()[0]
        width_size = x_.size()[3]

        # n * w * h * c
        x_ = x_.permute(0, 3, 2, 1)
        # n * w * hc
        x_ = x_.contiguous().view(batch_size, width_size, -1)        

        # n * w * h * c
        g_x = g_x.permute(0, 3, 2, 1)
        ori_size = g_x.size()
        # n * w * hc
        g_x = g_x.contiguous().view(batch_size, width_size, -1)        

        # n * w * h * c
        x_t = x_t.permute(0, 3, 2, 1)
        x_t = x_t.contiguous().view(batch_size, width_size, -1)       
        # n * hc * w
        x_t = x_t.permute(0, 2, 1)

        # n * w * hc | n * hc * w
        attention_map = torch.matmul(x_, x_t)
        attention_map = F.softmax(attention_map, dim = -1)

        out = torch.matmul(attention_map, g_x)
        # n * w * h * c
        out = out.view(batch_size, width_size, *ori_size[2:])
        out = out.permute(0, 3, 2, 1)

        return out      

class vectorAttBlockHeight(nn.Module):

    def __init__(self):

        super(vectorAttBlockHeight, self).__init__()

    def forward(self, x_, x_t, g_x):

        # n * c * h * w
        batch_size = x_.size()[0]
        height_size = x_.size()[2]

        # n * h * c * w
        x_ = x_.permute(0, 2, 1, 3)
        # n * h * cw
        x_ = x_.contiguous().view(batch_size, height_size, -1)        

        # n * h * c * w
        g_x = g_x.permute(0, 2, 1, 3)
        ori_size = g_x.size()
        # n * h * cw
        g_x = g_x.contiguous().view(batch_size, height_size, -1)        

        # n * h * c * w
        x_t = x_t.permute(0, 2, 1, 3)
        x_t = x_t.contiguous().view(batch_size, height_size, -1)       
        # n * cw * h
        x_t = x_t.permute(0, 2, 1)

        # n * h * cw | n * cw * h
        attention_map = torch.matmul(x_, x_t)
        attention_map = F.softmax(attention_map, dim = -1)

        out = torch.matmul(attention_map, g_x)
        # n * w * h * c
        out = out.view(batch_size, height_size, *ori_size[2:])
        out = out.permute(0, 2, 1, 3)

        return out        

class vectorAttBlockChannel(nn.Module):

    def __init__(self):

        super(vectorAttBlockChannel, self).__init__()

    def forward(self, x_, x_t, g_x):

        # n * c * h * w
        batch_size = x_.size()[0]
        channel_size = x_.size()[1]

        # n * c * hw
        x_ = x_.contiguous().view(batch_size, channel_size, -1)        

        # n * c * h * w
        ori_size = g_x.size()
        # n * c * hw
        g_x = g_x.contiguous().view(batch_size, channel_size, -1)        

        # n * c * hw
        x_t = x_t.contiguous().view(batch_size, channel_size, -1)       
        # n * hw * c
        x_t = x_t.permute(0, 2, 1)

        # n * c * hw | n * hw * c
        attention_map = torch.matmul(x_, x_t)
        attention_map = F.softmax(attention_map, dim = -1)

        out = torch.matmul(attention_map, g_x)
        # n * c * h * w
        out = out.view(batch_size, channel_size, *ori_size[2:])

        return out      