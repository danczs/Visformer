#mg_size=224, init_channels=32, embed_dim=384, depth=[1,8,14,2], num_heads=[2,4,8,16],
img_size=224
patch_size=16
init_channels=32
num_classes=1000
embed_dim=384
depth=[1,8,14,2]
num_heads=[2,4,8,16]
# depth=[0,7,4,4]
# num_heads=[6,6,6,6]
mlp_ratio=4
pos_embed=True
group=8

flops = {}
params = {}

#stem
feature_size = img_size // 2 #112
stem_params = 3 * init_channels * 7 * 7
stem_flops = stem_params * feature_size * feature_size
params['stem'] = stem_params
flops['stem'] = stem_flops

#pe0
feature_size = feature_size // 2 #56
pe0_params = embed_dim // 4 * init_channels * 2 * 2
pe0_flops = pe0_params * feature_size * feature_size
params['pe0'] = pe0_params
flops['pe0'] = pe0_flops

#stage 0
s0_c1_params = embed_dim // 4 * embed_dim // 4 * 2 * 1 * 1
s0_c2_params = 3 * 3 * embed_dim // 2 * embed_dim // 2 // 8
s0_c3_params = embed_dim // 4 * embed_dim // 4 * 2 * 1 * 1
s0_b_params = s0_c1_params + s0_c2_params + s0_c3_params
s0_b_flops = s0_b_params * feature_size * feature_size
for b in range(depth[0]):
    params['s0_b{}'.format(b)] = s0_b_params
    flops['s0_b{}'.format(b)] = s0_b_flops

#pe1
feature_size = feature_size // 2 #28
pe1_params = embed_dim // 4 * embed_dim // 2 * 2 * 2
pe1_flops = pe1_params * feature_size * feature_size
params['pe1'] = pe1_params
flops['pe1'] = pe1_flops

#stage1
s1_c1_params = embed_dim // 2 * embed_dim // 2 * 2 * 1 * 1
s1_c2_params = 3 * 3 * embed_dim * embed_dim // 8
s1_c3_params = embed_dim // 2 * embed_dim // 2 * 2 * 1 * 1
s1_b_params = s1_c1_params + s1_c2_params + s1_c3_params
s1_b_flops = s1_b_params * feature_size * feature_size
for b in range(depth[1]):
    params['s1_b{}'.format(b)] = s1_b_params
    flops['s1_b{}'.format(b)] = s1_b_flops

#pe2
feature_size = feature_size // 2 #14
pe2_params = embed_dim // 2 * embed_dim * 2 * 2
pe2_flops = pe2_params * feature_size * feature_size
params['pe2'] = pe2_params
flops['pe2'] = pe2_flops

#stage2
attn_qkv_params = embed_dim * embed_dim * 1 * 1 * 3
attn_proj_params = embed_dim * embed_dim * 1 * 1
attn_params = attn_qkv_params + attn_proj_params
h = num_heads[2]
token_num = feature_size * feature_size
attn_score_flops = (token_num * token_num *  embed_dim // h + \
                token_num * token_num * embed_dim // h ) * h
attn_flops = attn_params * feature_size * feature_size + attn_score_flops

mlp_params = 1 * 1 * embed_dim * embed_dim * mlp_ratio * 2
mlp_flops = mlp_params * feature_size * feature_size
for b in range(depth[2]):
    params['s2_attn_b{}'.format(b)] = attn_params
    params['s2_mlp_b_{}'.format(b)] = mlp_params
    flops['s2_attn_b{}'.format(b)] = attn_flops
    flops['s2_mlp_b_{}'.format(b)] = mlp_flops

#pe3
feature_size = feature_size // 2 #7
pe3_params = embed_dim * embed_dim * 2 * 2 * 2
pe3_flops = pe3_params * feature_size * feature_size
params['pe3'] = pe3_params
flops['pe3'] = pe3_flops

#stage3
embed_dim = embed_dim * 2
attn_qkv_params = embed_dim * embed_dim * 1 * 1 * 3
attn_proj_params = embed_dim * embed_dim * 1 * 1
attn_params = attn_qkv_params + attn_proj_params
h = num_heads[3]
token_num = feature_size * feature_size
attn_score_flops = (token_num * token_num * embed_dim // h + \
                token_num * token_num * embed_dim // h ) * h
attn_flops = attn_params * feature_size * feature_size + attn_score_flops

mlp_params = 1 * 1 * embed_dim * embed_dim * mlp_ratio * 2
mlp_flops = mlp_params * feature_size * feature_size
for b in range(depth[3]):
    params['s3_attn_b{}'.format(b)] = attn_params
    params['s3_mlp_b_{}'.format(b)] = mlp_params
    flops['s3_attn_b{}'.format(b)] = attn_flops
    flops['s3_mlp_b_{}'.format(b)] = mlp_flops

#cls
cls_params = embed_dim * num_classes
cls_flops = cls_params
params['cls'] = cls_params
flops['cls'] = cls_flops

all_flops = sum(flops.values())
all_params = sum(params.values())

print(all_flops)
print(all_params)
for k in params.keys():
    print(k+':',flops[k],'{}%'.format(round(100 * flops[k] / all_flops,1)), params[k], '{}%'.format(round(100* params[k] / all_params,1)))

# print(params)
# print(),sum())


