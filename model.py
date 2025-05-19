from Code.lib.Swin import SwinTransformer
from torch import nn, Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
# from modules import DeformConv
import copy
from Code.lib.IHN.network import IHN
from options import opt
import random
from Code.lib.IHN.utils import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def count_zeros(tensor):
    return (tensor == 0).sum().item()

def generate_t_hat_map(t_hat):
    t_hat_map = (t_hat != 0).float()
    t_hat_map = torch.sum(t_hat_map, dim=1, keepdim=True)
    t_hat_map = (t_hat_map > 0).float()
    
    return t_hat_map

#model
class TMSOD(nn.Module):
    def __init__(self):
        super(TMSOD, self).__init__()
        # self.depth = nn.Conv2d(1, 3, kernel_size=1)
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.IHN = IHN(args=opt)
        for param in self.IHN.parameters():
            param.requires_grad = False

        for n, value in self.IHN.fnet1.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
                print("False")
                print(n)
            else:
                value.requires_grad = True
                print("True")
                print(n)

        self.IHN.eval()

        self.MSA_sem = GMSA_ini(d_model=1024)
        self.conv_sem = conv3x3_bn_relu(1024*2, 1024)

        self.conv_sem3 = nn.Sequential(
            conv3x3(1024, out_planes=1024, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            )
        self.conv_sem2 = nn.Sequential(
            conv3x3(1024, out_planes=512, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )
        self.conv_sem1 = nn.Sequential(
            conv3x3(1024, out_planes=256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )

        self.MSA4_r = GMSA_ini(d_model=1024)
        self.MSA3_r = GMSA_ini(d_model=512)
        self.MSA2_r = GMSA_ini(d_model=256)

        self.MSA4_2_r = GMSA_ini(d_model=1024)
        self.MSA3_2_r = GMSA_ini(d_model=512)
        self.MSA2_2_r = GMSA_ini(d_model=256)

        self.conv_attr4 = conv3x3_bn_relu(1024, 1024)
        self.conv_attr3 = conv3x3_bn_relu(512, 512)
        self.conv_attr2 = conv3x3_bn_relu(256, 256)

        self.conv_fr4 = conv3x3_bn_relu(1024, 1024)
        self.conv_fr3 = conv3x3_bn_relu(512, 512)
        self.conv_fr2 = conv3x3_bn_relu(256, 256)

        self.convAtt4 = conv3x3_bn_relu(1024, 1024)
        self.convAtt3 = conv3x3_bn_relu(512, 512)
        self.convAtt2 = conv3x3_bn_relu(256, 256)
        self.convAtt1 = conv3x3_bn_relu(128, 128)

        self.conv1024 = conv3x3_bn_relu(1024, 512)
        self.conv512 = conv3x3_bn_relu(512, 256)
        self.conv256 = conv3x3_bn_relu(256, 128)
        self.conv128 = conv3x3_bn_relu(128, 64)
        self.conv64 = conv3x3(64, 1)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)


    def forward(self, rgb, t, img1, img2):
        fr = self.rgb_swin(rgb)#[0-3]
        ft = self.t_swin(t)

        semantic_warp = self.MSA_sem(torch.cat((fr[3].flatten(2).transpose(1, 2), ft[3].flatten(2).transpose(1, 2)), dim=1), torch.cat((fr[3].flatten(2).transpose(1, 2), ft[3].flatten(2).transpose(1, 2)), dim=1))#(b,c,hw)->(b,hw,c), cat in hw for att, which contains self and cross.
        semantic1, semantic2 = torch.split(semantic_warp, fr[3].shape[2] * fr[3].shape[3], dim=1)
        semantic_warp = self.conv_sem(torch.cat((semantic1.view(semantic1.shape[0], int(np.sqrt(semantic1.shape[1])), int(np.sqrt(semantic1.shape[1])), -1).permute(0, 3, 1, 2).contiguous(), semantic2.view(semantic2.shape[0], int(np.sqrt(semantic2.shape[1])), int(np.sqrt(semantic2.shape[1])), -1).permute(0, 3, 1, 2).contiguous()), dim=1))

        semantic_warp3 = self.conv_sem3(self.up1(semantic_warp))
        semantic_warp2 = self.conv_sem2(self.up2(semantic_warp))
        semantic_warp1 = self.conv_sem1(self.up4(semantic_warp))

        four_point_disp, coords1, coords0 = self.IHN(img1, img2, semantic_warp)
        flow_med = coords1 - coords0
        flow_med = F.upsample_bilinear(flow_med, None, [12, 12]) * 12
        t_hat = warp(t, flow_med)

        t_hat_mask = generate_t_hat_map(t_hat)

        fr1_masked = fr[1] * F.interpolate(t_hat_mask, size=fr[1].size()[2:], mode='bilinear', align_corners=False)
        fr2_masked = fr[2] * F.interpolate(t_hat_mask, size=fr[2].size()[2:], mode='bilinear', align_corners=False)
        fr3_masked = fr[3] * F.interpolate(t_hat_mask, size=fr[3].size()[2:], mode='bilinear', align_corners=False)

        att_4_r = self.MSA4_r((fr3_masked * semantic_warp3).flatten(2).transpose(1, 2), ft[3].flatten(2).transpose(1, 2))
        att_3_r = self.MSA3_r((fr2_masked * semantic_warp2).flatten(2).transpose(1, 2), ft[2].flatten(2).transpose(1, 2))
        att_2_r = self.MSA2_r((fr1_masked * semantic_warp1).flatten(2).transpose(1, 2), ft[1].flatten(2).transpose(1, 2))
        r1 = self.convAtt1(fr[0])
        
        fr4 = self.conv_fr4(self.conv_attr4(att_4_r.view(att_4_r.shape[0], int(np.sqrt(att_4_r.shape[1])), int(np.sqrt(att_4_r.shape[1])), -1).permute(0, 3, 1, 2).contiguous()) + fr[3])
        fr3 = self.conv_fr3(self.conv_attr3(att_3_r.view(att_3_r.shape[0], int(np.sqrt(att_3_r.shape[1])), int(np.sqrt(att_3_r.shape[1])), -1).permute(0, 3, 1, 2).contiguous()) + fr[2])
        fr2 = self.conv_fr2(self.conv_attr2(att_2_r.view(att_2_r.shape[0], int(np.sqrt(att_2_r.shape[1])), int(np.sqrt(att_2_r.shape[1])), -1).permute(0, 3, 1, 2).contiguous()) + fr[1])

        att_4_r = self.MSA4_2_r(fr4.flatten(2).transpose(1, 2), fr4.flatten(2).transpose(1, 2))
        att_3_r = self.MSA3_2_r(fr3.flatten(2).transpose(1, 2), fr3.flatten(2).transpose(1, 2))
        att_2_r = self.MSA2_2_r(fr2.flatten(2).transpose(1, 2), fr2.flatten(2).transpose(1, 2))

        r4 = self.convAtt4(att_4_r.view(att_4_r.shape[0], int(np.sqrt(att_4_r.shape[1])), int(np.sqrt(att_4_r.shape[1])), -1).permute(0, 3, 1, 2).contiguous())
        r3 = self.convAtt3(att_3_r.view(att_3_r.shape[0], int(np.sqrt(att_3_r.shape[1])), int(np.sqrt(att_3_r.shape[1])), -1).permute(0, 3, 1, 2).contiguous())
        r2 = self.convAtt2(att_2_r.view(att_2_r.shape[0], int(np.sqrt(att_2_r.shape[1])), int(np.sqrt(att_2_r.shape[1])), -1).permute(0, 3, 1, 2).contiguous())

        r4 = self.conv1024(self.up2(r4))
        r3 = self.conv512(self.up2(r3 + r4))
        r2 = self.conv256(self.up2(r2 + r3))
        r1 = self.conv128(r1 + r2)
        out = self.up4(r1)
        out = self.conv64(out)
        return out

    def load_pre(self, pre_model, IHN):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.t_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")
        self.IHN.load_state_dict(torch.load(IHN, map_location='cuda:0'), strict=False)
        print(f"IHN loading pre_model ${IHN}")

class GMSA_ini(nn.Module):
    def __init__(self, d_model=256, num_layers=2, decoder_layer=None):
        super(GMSA_ini, self).__init__()
        if decoder_layer is None:
            decoder_layer = GMSA_layer_ini(d_model=d_model, nhead=8)
        self.layers = _get_clones(decoder_layer, num_layers)
    def forward(self, fr, ft):
        # fr = fr.flatten(2).transpose(1, 2)  # b hw c
        # ft = ft.flatten(2).transpose(1, 2)
        output = fr
        for layer in self.layers:
            output = layer(output, ft)
        return output
class GMSA_layer_ini(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(GMSA_layer_ini, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.sigmoid = nn.Sigmoid()
    def forward(self, fr, ft, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):


        fr2 = self.multihead_attn(query=self.with_pos_embed(fr, query_pos).transpose(0, 1),#hw b c
                                   key=self.with_pos_embed(ft, pos).transpose(0, 1),
                                   value=ft.transpose(0, 1))[0].transpose(0, 1)#b hw c
        fr = fr + self.dropout2(fr2)
        fr = self.norm2(fr)

        fr2 = self.linear2(self.dropout(self.activation(self.linear1(fr))))  #FFN
        fr = fr + self.dropout3(fr2)
        fr = self.norm3(fr)
        # print(fr.shape)
        return fr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

#gated MSA
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# gated MSA layer
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")