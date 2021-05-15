import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from model import common


def make_model(args):
    return MODEL(args)

class PAF(nn.Module): # path attention fusion
    def __init__(self, conv, channel, n_feats, reduction=16):
        super(PAF, self).__init__()
   
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.pathfusion = conv(channel, n_feats, 1)

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x

        y = self.avg_pool(x)
        y = self.conv_du(y)

        x = self.pathfusion(x * y)
        
        return x 


class BasicBlock(nn.Module):  # AU and DPB
    def __init__(
            self, conv, in_chs, dpag_feats, res_chs, des_chs, block_type='body'):
        super(BasicBlock, self).__init__()

        self.res_chs = res_chs
        
        if block_type is 'head':
            self.head = True  
        else:
            assert block_type is 'body'
            self.head = False

        if self.head:
            self.allocation = conv(in_chs, res_chs + des_chs, 1)

        # dual path block   
        self.DPB = nn.Sequential(*[
            conv(in_chs, dpag_feats, 1),
            nn.ReLU(True),
            conv(dpag_feats, res_chs + des_chs, 3)
        ])



    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x

        if self.head:         
            x_s = self.allocation(x_in)
            x_res = x_s[:, :self.res_chs, :, :]  # residual imformation flow
            x_dense = x_s[:, self.res_chs:, :, :]
        else:
            x_res = x[0]
            x_dense = x[1]
    
        x_in = self.DPB(x_in)    
        res = x_in[:, :self.res_chs, :, :]
        dense = x_in[:, self.res_chs:, :, :]
        x_res = x_res + res
        x_dense = torch.cat([x_dense, dense], dim=1)
        return x_res, x_dense


class DPAG(nn.Module):  # DPAG
    def __init__(self, conv, n_feats, in_chs, dpag_feats, res_chs, des_chs, n_dpblocks):
        super(DPAG, self).__init__()

        # define the weight of the global skip connection
        self.belta = nn.Parameter(torch.FloatTensor([0]))
        # define allocation unit
        head = []
        head.append(BasicBlock(conv, in_chs, dpag_feats,  res_chs, des_chs, 'head'))

        # define dual path units
        in_chs = res_chs + 2 * des_chs
        body = []
        for i in range(n_dpblocks):
            body.append(BasicBlock(conv, in_chs, dpag_feats,  res_chs, des_chs, 'body'))
            in_chs += des_chs
     
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = PAF(conv, in_chs, n_feats)
        
    def forward(self, x, F_0):

        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        
        x = res + x + self.belta * F_0
        return x


class MODEL(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
       
        # define shallow features extraction 
        self.SF = conv(args.n_colors, n_feats, 3)


        # define SDPG      
        self.n_dpagroups = args.n_dpagroups
        res_chs = args.resChs 
        des_chs = args.denseChs  
        dpag_feats = args.dpag_feats 
        n_dpblocks = args.n_dpblocks  
        in_chs = n_feats

     
        self.SDPG = nn.ModuleList()
        for i in range(self.n_dpagroups):
            self.SDPG.append(
                DPAG(conv, n_feats, in_chs, dpag_feats, res_chs, des_chs, n_dpblocks)
            )
        
        self.SDPG_conv = conv(n_feats, n_feats, 3)

        self.upsample = nn.Sequential(*[
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ])
      
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)


    def forward(self, x):
 
        x = self.sub_mean(x)
  
        x = self.SF(x)  # shallow feature extract     
        F_0 = x

        for i in range(self.n_dpagroups):
            x = self.SDPG[i](x, F_0)
        x = self.SDPG_conv(x)
        
        x = x + F_0
        x = self.upsample(x)
        
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('upsample') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('upsample') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
