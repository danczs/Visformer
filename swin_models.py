import torch
import torch.nn as nn
from einops import rearrange
from weight_init import to_2tuple, trunc_normal_
import torch.nn.functional as F

__all__=[
    'swin_visformer_small_v2', 'swin_visformer_tiny_v2'
]

def drop_path(x, drop_prob:float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.LayerNorm):
    """ Layernorm with BCHW tensors """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=1, keepdim=True)
        x = ( x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
        return x



class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1, track_running_stats=True)

    def forward(self, x):
        return self.bn(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., group=8, spatial_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            if group < 2: #net setting
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2
        self.hidden_features = hidden_features
        self.group = group
        self.drop = nn.Dropout(drop)
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, bias=False)
        self.act1 = act_layer()
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, stride=1, padding=1,
                                   groups=self.group, bias=False)
            self.act2 = act_layer()
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)

        if self.spatial_conv:
            x = self.conv2(x)
            x = self.act2(x)

        x = self.conv3(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

#swin_transformer in BCHW format
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., head_dim_ratio=1.0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = int(dim // num_heads * head_dim_ratio )
        #self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)
        qk_scale_factor = qk_scale if qk_scale is not None else -0.25
        self.scale = self.head_dim ** qk_scale_factor
        self.qkv = nn.Conv2d(dim ,self.head_dim * num_heads * 3, 1, stride=1, padding=0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        B_, C, H, W = x.shape
        x = self.qkv(x)
        qkv = rearrange(x, 'b (x y z) h w -> x b y (h w) z', x=3, y=self.num_heads, z=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ( (q * self.scale) @ (k.transpose(-2,-1) * self.scale))

        ###
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            N = H * W
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'b y (h w) z -> b (y z) h w', h=H, w=W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, head_dim_ratio=1., mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm,
                 group=8, attn_disabled=False, spatial_conv=False):
        super().__init__()
        self.attn_disabled = attn_disabled
        self.spatial_conv = spatial_conv
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if not attn_disabled:
            self.norm1 = norm_layer(dim)
            #self.attn = Attention(dim, num_heads=num_heads, head_dim_ratio=head_dim_ratio, qkv_bias=qkv_bias,
            #                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                        head_dim_ratio=head_dim_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       group=group, spatial_conv=spatial_conv) # new setting

        if not self.attn_disabled and self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros( (1, 1, H, W))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)


    def forward(self, x):
        B, C, H, W = x.shape
        if not self.attn_disabled:
            shortcut = x
            x = self.norm1(x)
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2,3))
            else:
                shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)

            # merge windows
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B C H' W'

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            else:
                x = shifted_x

            # FFN
            x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_pe = norm_layer is not None
        if self.norm_pe:
            self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) does not match model ({self.img_size[1]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.norm_pe:
            x = self.norm(x)
        return x


class SwinVisformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, init_channels=32, num_classes=1000, embed_dim=384, depth=[0,7,4,4],
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=LayerNorm, attn_stage='111', pos_embed=True, spatial_conv='111',
                 group=8, pool=True, conv_init=False, embedding_norm=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.pool = pool
        self.conv_init = conv_init
        assert ( isinstance(depth, list) or isinstance(depth, tuple) ) and len(depth) == 4
        if not (isinstance(num_heads, list) or isinstance(num_heads, tuple) ) :
            num_heads = [num_heads] * 4



        self.pos_embed = pos_embed
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        #stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.init_channels, 7, stride=2, padding=3, bias=False),
            BatchNorm(self.init_channels),
            nn.ReLU(inplace=True)
        )
        img_size //= 2

        # stage 0
        self.patch_embed0 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=self.init_channels,
                                       embed_dim=embed_dim // 4, norm_layer=embedding_norm)
        img_size //= 2
        if self.pos_embed:
            self.pos_embed0 = nn.Parameter(torch.zeros(1, embed_dim//4, img_size, img_size))
            self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage0 = nn.ModuleList([
            Block(
                dim=embed_dim//4, input_resolution=(img_size,img_size), window_size=56,
                shift_size= 0 if (i%2 == 0 ) else 28, num_heads=num_heads[0], head_dim_ratio=0.25,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group=group,
                attn_disabled= (attn_stage[0]=='0'), spatial_conv=(spatial_conv[0]=='1')
            )
            for i in range(0, sum(depth[:1]))
        ])

        #stage 1
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim//4,
                                       embed_dim=embed_dim//2, norm_layer=embedding_norm)
        img_size //= 2
        if self.pos_embed:
            self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim//2, img_size, img_size))

        self.stage1 = nn.ModuleList([
            Block(
                dim=embed_dim // 2, input_resolution=(img_size, img_size), window_size=28,
                shift_size=0 if (i % 2 == 0) else 14, num_heads=num_heads[1], head_dim_ratio=0.5,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,drop_path=dpr[i], norm_layer=norm_layer, group=group,
                attn_disabled= (attn_stage[1]=='0'), spatial_conv=(spatial_conv[1]=='1')
            )
            for i in range(sum(depth[:1]), sum(depth[:2]))
        ])

        #stage2

        self.patch_embed2 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim//2, embed_dim=embed_dim,
                                       norm_layer=embedding_norm)
        img_size //= 2
        if self.pos_embed:
            self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, img_size, img_size))

        self.stage2 = nn.ModuleList([
            Block(
                dim=embed_dim, input_resolution=(img_size, img_size), window_size=14,
                shift_size=0 if (i % 2 == 0) else 7, num_heads=num_heads[2], head_dim_ratio=1.0,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group=group,
                attn_disabled=(attn_stage[2] == '0'), spatial_conv=(spatial_conv[2] == '1')
            )
            for i in range(sum(depth[:2]), sum(depth[:3]))
        ])

        # stage 3

        self.patch_embed3 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim*2,
                                       norm_layer=embedding_norm)
        img_size //= 2
        if self.pos_embed:
            self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim*2, img_size, img_size))
        self.stage3 = nn.ModuleList([
            Block(
                dim=embed_dim*2, input_resolution=(img_size, img_size), window_size=7,
                shift_size=0 if (i % 2 == 0) else 3, num_heads=num_heads[3], head_dim_ratio=1.0,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group=group,
                attn_disabled=(attn_stage[3] == '0'), spatial_conv=(spatial_conv[3] == '1')
            )
            for i in range(sum(depth[:3]), sum(depth[:4]))
        ])

        # head
        if self.pool:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.norm = norm_layer(embed_dim*2)
        self.head = nn.Linear(embed_dim*2, num_classes)

        # weights init
        if self.pos_embed:
            trunc_normal_(self.pos_embed0, std=0.02)
            trunc_normal_(self.pos_embed1, std=0.02)
            trunc_normal_(self.pos_embed2, std=0.02)
            trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):

        x = self.stem(x)

        #stage 0
        x = self.patch_embed0(x)
        if self.pos_embed:
            x = x + self.pos_embed0
            x = self.pos_drop(x)
        for b in self.stage0:
            x = b(x)

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = x + self.pos_embed1
            x = self.pos_drop(x)
        for b in self.stage1:
            x = b(x)

        # stage 2
        x = self.patch_embed2(x)
        if self.pos_embed:
            x = x + self.pos_embed2
            x = self.pos_drop(x)
        for b in self.stage2:
            x = b(x)

        # stage3
        x = self.patch_embed3(x)
        if self.pos_embed:
            x = x + self.pos_embed3
            x = self.pos_drop(x)
        for b in self.stage3:
            x = b(x)

        # head
        x = self.norm(x)
        if self.pool:
            x = self.global_pooling(x)
        else:
            x = x[:, :, 0, 0]

        x = self.head( x.view(x.size(0), -1) )
        return x


def visformer_tiny(**kwargs):
    model = SwinVisformer(img_size=224, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, **kwargs)
    return model


def swin_visformer_small(**kwargs):
    model = SwinVisformer(img_size=224, init_channels=32, embed_dim=384, depth=[0,7,4,4], num_heads=6, mlp_ratio=4.,
                      group=8, attn_stage='0011', spatial_conv='1100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, **kwargs)
    return model

def swin_visformer_small_v2(**kwargs):
    model = SwinVisformer(img_size=224, init_channels=32, embed_dim=256, depth=[1,10,14,3], num_heads=[2,4,8,16],
                      mlp_ratio=4., group=8, attn_stage='0011', spatial_conv='1100', norm_layer=BatchNorm,
                      conv_init=True, embedding_norm=BatchNorm, **kwargs)
    return model

def swin_visformer_tiny_v2(**kwargs):
    model = SwinVisformer(img_size=224, init_channels=24, embed_dim=192, depth=[1, 4, 6, 3], num_heads=[1, 3, 6, 12],
                          mlp_ratio=4., group=8, attn_stage='0011', spatial_conv='1100', norm_layer=BatchNorm,
                          conv_init=True, embedding_norm=BatchNorm, **kwargs)
    return model

if __name__ == '__main__':
    torch.manual_seed(0)
    inputs = torch.rand(2, 3, 224, 224)

    net = swin_visformer_small_v2()

    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of parameters:{}'.format(parameters))
    x = net(inputs)
    print(x.shape)





