# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False, global_align=False, patch_size=16):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2) * 1e-4 / kernel_size,
            requires_grad=True,
        )
        self.weight_bias = torch.zeros(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2).cuda()
        self.weight_bias[:,:,:,:,:,kernel_size**2//2] = 1
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1]), requires_grad=True
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.global_align = global_align
        self.patch_size = patch_size
        if global_align:
            self.global_align_weight = nn.Parameter(
                torch.randn(1, 1, output_size[0]*output_size[1]//(self.patch_size**2), output_size[0]*output_size[1]//(self.patch_size**2)) * 1e-4 / self.patch_size,
                requires_grad=True,
            )
            self.global_align_weight_bias = torch.eye(output_size[0]*output_size[1]//(self.patch_size**2))[None, None].cuda()

    def forward(self, x):

        if self.global_align:
            fmri_mask = x != 0
            _, c, h, w = x.size()
            # x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            x = x.view(_, c, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size).transpose(-2,-3)
            # x : _ c h//p w//p p p
            x = x.contiguous().view(_, c, -1, self.patch_size**2)
            # x : _ c hw//(p**2) p**2

            global_weight = self.global_align_weight + self.global_align_weight_bias
            x = global_weight @ x
            # x : _ c hw//(p**2) p**2

            x = x.view(_, c, h//self.patch_size, w//self.patch_size, self.patch_size, self.patch_size).transpose(-2, -3)
            x = x.contiguous().view(_, c, h, w) * fmri_mask


        fmri_mask = x != 0
        x = torch.nn.functional.pad(x, (self.kernel_size[0]//2,)*4)

        _, c, h, w = x.size()
        
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        # out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        conv_weight = self.weight_bias + self.weight
        out = (x.unsqueeze(1) * (conv_weight.abs() / conv_weight.norm(p=1,dim=-1,keepdim=True))).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out * fmri_mask