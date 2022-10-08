#from  import create_conv2d, DropPath, make_divisible, create_act_layer, get_norm_act_layer
from timm.models.efficientnet_blocks import DepthwiseSeparableConv, InvertedResidual,SqueezeExcite
from timm.models.layers import create_conv2d, create_classifier, get_norm_act_layer
from timm.models.efficientnet_builder import decode_arch_def, efficientnet_init_weights, round_channels, resolve_bn_args

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
#from .layers import
from timm.models.efficientnet import mobilenetv2_100
from swin_models import Block as SABlock


class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1, track_running_stats=True)

    def forward(self, x):
        return self.bn(x)

class BaseNet(nn.Module):
    def __init__(self, arch_def, stem_size=32, num_features=1280, num_classes=1000, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.stem = nn.Sequential(
            create_conv2d(3, stem_size, 3, stride=2, padding=''),
            norm_layer(stem_size),
            act_layer()
        )
        # self.conv_stem =
        # self.bn1 = norm_layer(stem_size)
        # self.relu1 = act_layer()
        self.blocks.append(self.stem)
        c_in = stem_size
        for blocks in arch_def:
            #
            block = blocks[0]
            #print(block, block[:2])
            if block[:2] == 'ds':
                btype,r,kernel,stride,c_out = block.split('_')
                r = int(r[1:])
                kernel = int(kernel[1:])
                stride = int(stride[1:])
                c_out = int(c_out[1:])

                for i in range(r):
                    stride = stride if i==0 else 1
                    b = DepthwiseSeparableConv(c_in,c_out,dw_kernel_size=kernel,stride=stride,
                                               act_layer=act_layer,norm_layer=norm_layer)
                    self.blocks.append(b)
                    c_in = c_out



            elif block[:2] == 'ir':
                #print(ir)
                btype, r, kernel, stride, e, c_out = block.split('_')
                r = int(r[1:])
                kernel = int(kernel[1:])
                stride = int(stride[1:])
                e = int(e[1:])
                c_out = int(c_out[1:])

                for i in range(r):
                    stride = stride if i == 0 else 1
                    b = InvertedResidual(c_in,c_out,dw_kernel_size=kernel,stride=stride,exp_ratio=e,
                                         act_layer=act_layer,norm_layer=norm_layer)
                    self.blocks.append(b)
                    c_in = c_out

            elif block[:2] == 'sa':
                btype, r, kernel, stride, e, c_out = block.split('_')
                r = int(r[1:])
                kernel = int(kernel[1:])
                stride = int(stride[1:])
                e = int(e[1:])
                c_out = int(c_out[1:])
                for i in range(r):
                    if i == 0 and stride == 2:
                        b = InvertedResidual(c_in, c_out, dw_kernel_size=kernel, stride=stride, exp_ratio=e,
                                             act_layer=act_layer, norm_layer=norm_layer)
                        self.blocks.append(b)
                        c_in = c_out
                    else:
                        window_size = 14 if c_out == 64 else 7
                        num_heads = c_out // 16
                        input_resolution = window_size
                        qk_scale = -0.5
                        b = SABlock(c_out, input_resolution=(window_size,window_size), num_heads=num_heads, window_size=window_size,
                                    shift_size=0, head_dim_ratio=1., mlp_ratio=e, qkv_bias=False, qk_scale=qk_scale,
                                    act_layer=nn.GELU, norm_layer=BatchNorm, attn_disabled=False, spatial_conv=False)
                        self.blocks.append(b)
                        c_in = c_out


        # Head + Pooling
        self.conv_head = create_conv2d(c_in, num_features, 1, padding='')
        self.bn_head = norm_layer(num_features)
        self.act_head = act_layer()
        self.global_pool, self.classifier = create_classifier(num_features, num_classes, pool_type='avg')
        efficientnet_init_weights(self)

    def forward(self,x):
        for b in self.blocks:
            x = b(x)
            #print(x.size())
        x = self.conv_head(x)
        x = self.bn_head(x)
        x = self.act_head(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def my_mobilenet_v2():
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]

    stem_size = 32
    return BaseNet(arch_def,stem_size=stem_size)

def vis_mobilenet_v2():
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r1_k3_s2_e6_c64'],
        ['sa_r6_k3_s1_e4_c64'],
        # ['ir_r3_k3_s1_e6_c96'],
        # ['sa_r3_k3_s1_e4_c96'],
        ['ir_r1_k3_s2_e6_c128'],
        ['sa_r1_k3_s1_e4_c128'],
        ['ir_r1_k3_s1_e6_c256'],
        ['sa_r1_k3_s1_e4_c256'],
    ]

    stem_size = 32
    return BaseNet(arch_def,stem_size=stem_size)

def timm_mobilenet_v2():
    return mobilenetv2_100()


def get_mobilenet_flops(arch_def, img_size = 224, stem_size=32, num_features=1280, num_classes=1000):
    flops = {}
    params = {}
    features_size = img_size // 2
    stem_params = 3 * stem_size *  3 * 3
    stem_flops = stem_params * features_size * features_size

    params['stem'] = stem_params
    flops['stem'] = stem_flops

    c_in = stem_size
    for blocks in arch_def:
        #
        block = blocks[0]
        # print(block, block[:2])
        if block[:2] == 'ds':
            btype, r, kernel, stride, c_out = block.split('_')
            r = int(r[1:])
            kernel = int(kernel[1:])
            stride = int(stride[1:])
            c_out = int(c_out[1:])

            for i in range(r):
                stride = stride if i == 0 else 1
                features_size = features_size // stride
                b_params = c_in * kernel * kernel + c_in * c_out
                b_flops = b_params * features_size * features_size
                params[block+'_{}'.format(i)] = b_params
                flops[block+'_{}'.format(i)] = b_flops

                c_in = c_out



        elif block[:2] == 'ir':
            # print(ir)
            btype, r, kernel, stride, e, c_out = block.split('_')
            r = int(r[1:])
            kernel = int(kernel[1:])
            stride = int(stride[1:])
            e = int(e[1:])
            c_out = int(c_out[1:])

            for i in range(r):
                stride = stride if i == 0 else 1

                #features_size = features_size // stride
                b_params = c_in * c_in * e
                b_flops = b_params * features_size * features_size
                features_size = features_size // stride
                b_params_2 = c_in * e * kernel * kernel + c_in * e * c_out
                b_params += b_params_2
                b_flops += b_params_2 * features_size * features_size
                params[block + '_{}'.format(i)] = b_params
                flops[block + '_{}'.format(i)] = b_flops
                c_in = c_out
                # b = InvertedResidual(c_in, c_out, dw_kernel_size=kernel, stride=stride, exp_ratio=e,
                #                      act_layer=act_layer, norm_layer=norm_layer)
                # self.blocks.append(b)
        elif block[:2] == 'sa':
            btype, r, kernel, stride, e, c_out = block.split('_')
            r = int(r[1:])
            kernel = int(kernel[1:])
            stride = int(stride[1:])
            e = int(e[1:])
            c_out = int(c_out[1:])
            for i in range(r):
                if i==0 and stride == 2:
                    b_params = c_in * c_in * e
                    b_flops = b_params * features_size * features_size
                    features_size = features_size // stride
                    b_params_2 = c_in * e * kernel * kernel + c_in * e * c_out
                    b_params += b_params_2
                    b_flops += b_params_2 * features_size * features_size
                    params[block + '_{}'.format(i)] = b_params
                    flops[block + '_{}'.format(i)] = b_flops
                    c_in = c_out
                else:
                    embed_dim = c_out
                    attn_qkv_params = embed_dim * embed_dim * 1 * 1 * 3
                    attn_proj_params = embed_dim * embed_dim * 1 * 1
                    attn_params = attn_qkv_params + attn_proj_params
                    h = 8#num_heads[3]
                    token_num = features_size * features_size
                    attn_score_flops = (token_num * token_num * embed_dim // h + \
                                        token_num * token_num * embed_dim // h) * h
                    attn_flops = attn_params * features_size * features_size + attn_score_flops

                    params[block + 'attn_{}'.format(i)] = attn_params
                    flops[block + 'attn_{}'.format(i)] = attn_flops

                    mlp_params = 1 * 1 * embed_dim * embed_dim * e * 2
                    mlp_flops = mlp_params * features_size * features_size
                    params[block + 'mlp_{}'.format(i)] = mlp_params
                    flops[block + 'mlp_{}'.format(i)] = mlp_flops

    # head
    print(features_size)
    head_params = c_in * num_features
    head_flops = head_params * features_size * features_size
    params['head'] = head_params
    flops['head'] = head_flops

    # cls
    cls_params = num_features * num_classes
    cls_flops = cls_params
    params['cls'] = cls_params
    flops['cls'] = cls_flops

    all_flops = sum(flops.values())
    all_params = sum(params.values())

    print(all_flops)
    print(all_params)
    for k in params.keys():
        print(k + ':', flops[k], '{}%'.format(round(100 * flops[k] / all_flops, 1)), params[k],
              '{}%'.format(round(100 * params[k] / all_params, 1)))

    # print(params)
    # print(),sum())


if __name__ == '__main__':
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    # arch_def = [
    #     ['ds_r1_k3_s1_c16'],
    #     ['ir_r2_k3_s2_e6_c24'],
    #     ['ir_r3_k3_s2_e6_c32'],
    #     ['ir_r1_k3_s2_e6_c64'],
    #     ['sa_r6_k3_s1_e4_c64'],
    #     #['ir_r3_k3_s1_e6_c96'],
    #     #['sa_r3_k3_s1_e4_c96'],
    #     ['ir_r1_k3_s2_e6_c128'],
    #     ['sa_r1_k3_s1_e4_c128'],
    #     ['ir_r1_k3_s1_e6_c256'],
    #     ['sa_r1_k3_s1_e4_c256'],
    # ]
    # arch_def = [
    #     ['ds_r1_k3_s1_c16'],
    #     ['ir_r2_k3_s2_e6_c24'],
    #     ['ir_r3_k3_s2_e6_c32'],
    #     ['ir_r1_k3_s2_e6_c96'],
    #     #['sa_r2_k3_s1_e4_c64'],
    #     #['ir_r1_k3_s1_e6_c96'],
    #     ['sa_r3_k3_s1_e4_c96'],
    #     ['ir_r1_k3_s2_e6_c128'],
    #     ['sa_r1_k3_s1_e4_c128'],
    #     ['ir_r1_k3_s1_e6_c256'],
    #     ['sa_r1_k3_s1_e4_c256'],
    # ]
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r1_k3_s2_e6_c64'],
        ['sa_r6_k3_s1_e4_c64'],
        # ['ir_r3_k3_s1_e6_c96'],
        # ['sa_r3_k3_s1_e4_c96'],
        ['ir_r1_k3_s2_e6_c128'],
        ['sa_r1_k3_s1_e4_c128'],
        ['ir_r1_k3_s1_e6_c256'],
        ['sa_r1_k3_s1_e4_c256'],
    ]

    stem_size = 32
    #torch.manual_seed(0)
    my_mobilenet_v2 = BaseNet(arch_def,stem_size=stem_size)
    num_params = sum(param.numel() for param in my_mobilenet_v2.parameters())
    print(num_params)
    print(my_mobilenet_v2)
    #for param in my_mobilenet_v2.parameters():
    #    print('ti:',param.size(),torch.mean(param))

    #torch.manual_seed(0)
    mobilenetv2 = mobilenetv2_100()
    num_params = sum(param.numel() for param in mobilenetv2.parameters())

    #for param in mobilenetv2.parameters():
    #    print('my:',param.size(),torch.mean(param))

    a = torch.randn(1,3,224,224)
    print(num_params)
    get_mobilenet_flops(arch_def)
    #print(mobilenetv2)
    b1 = my_mobilenet_v2(a)
    b2 = mobilenetv2(a)
