# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import random

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]




class MaskedVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def random_masking(self, x, mask_ratio): # mask_ratio: 버리는 비율
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand([N, L], device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_masked = torch.cat([cls_token, x_masked], dim=1)

        return x_masked

    def forward_features(self, x, block_index=None, drop_rate=0, mask_count = 0):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        # interpolate patch embeddings
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        class_pos_embed = self.pos_embed[:, 0]
        N = self.pos_embed.shape[1] - 1
        patch_pos_embed = self.pos_embed[:, 1:]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        if w0 != patch_pos_embed.shape[-2]:
            helper = torch.zeros(h0)[None, None, None, :].repeat(1, dim, w0 - patch_pos_embed.shape[-2], 1).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-2)
        if h0 != patch_pos_embed.shape[-1]:
            helper = torch.zeros(w0)[None, None, :, None].repeat(1, dim, 1, h0 - patch_pos_embed.shape[-1]).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-1)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        # interpolate patch embeddings finish

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)
        # ryu
        print(drop_rate)
        print("before ", x.shape)
        x = self.random_masking(x, mask_count//196)
        print("after  ", x.shape)

        layer_wise_tokens = []
        for idx, blk in enumerate(self.blocks):

            if block_index is not None and idx == block_index:
                token = x[:, :1, :]
                features = x[:, 1:, :]
                row = np.random.choice(range(x.shape[1] - 1), size=int(drop_rate*x.shape[1]), replace=False)
                features[:, row, :] = 0.0
                x = torch.cat((token, features), dim=1)

            x = blk(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]

        return [x[:, 0] for x in layer_wise_tokens], [x for x in layer_wise_tokens]

    def forward(self, x, block_index=None, drop_rate=0, patches=False, mask_count = 0):
        list_out, patch_out = self.forward_features(x, block_index, drop_rate, mask_count = mask_count)
        x = [self.head(x) for x in list_out]
        if patches:
            return x, patch_out
        else:
            return x
        

@register_model
def masked_deit_small_patch16_224(pretrained=False, **kwargs):
    model = MaskedVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#     def __init__(self, global_pool=False, **kwargs):
#         super(VisionTransformer, self).__init__(**kwargs)

#         self.global_pool = global_pool
#         if self.global_pool:
#             norm_layer = kwargs['norm_layer']
#             embed_dim = kwargs['embed_dim']
#             self.fc_norm = norm_layer(embed_dim)

#             del self.norm  # remove the original norm

#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         if self.global_pool:
#             x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             outcome = self.fc_norm(x)
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]

#         return outcome


# class augsub_VisionTransformer(VisionTransformer):

#     def __init__(self, **kwargs):
#         super(augsub_VisionTransformer, self).__init__(**kwargs)

#         # Do not use nn.Identity for all transformer blocks
#         for block in self.blocks:
#             if isinstance(block.drop_path, nn.Identity):
#                 block.drop_path = DropPath(0.)

#     def random_masking(self, x, mask_ratio): # mask_ratio: 버리는 비율
#         cls_token = x[:, :1, :]
#         x = x[:, 1:, :]
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))

#         noise = torch.rand([N, L], device=x.device)

#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#         x_masked = torch.cat([cls_token, x_masked], dim=1)

#         return x_masked

#     def forward_features(self, x, mask_ratio=0.0):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         # print(x.shape)
#         if mask_ratio > 0.0:
#             # print(x.shape)
#             x = self.random_masking(x, mask_ratio)
#             # print(x.shape)
#             # print("masked!")

#         x = self.pos_drop(x)
#         for blk in self.blocks:
#             x = blk(x)

#         if self.global_pool:
#             x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             outcome = self.fc_norm(x)
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]

#         return outcome

#     def forward(self, x, augsub_type='masking', augsub_ratio=0.0):
#         if augsub_type == 'dropout':
#             raise NotImplementedError('Augdrop is not implemented yet')
#         elif augsub_type == 'droppath':
#             raise NotImplementedError('Augpath is not implemented yet')
#         elif augsub_type == 'masking':
#             # x = self.forward_features(x, augsub_ratio)
#             feature_out = self.forward_features(x, augsub_ratio)
#         else:
#             # x = self.forward_features(x)
#             feature_out = self.forward_features(x)
#         x = self.head(feature_out)

#         # return x, feature_out
#         return x

# from timm.models.vision_transformer import _cfg





# def vit_base_patch16(pretrained=True, **kwargs):
#     model = augsub_VisionTransformer(
#         patch_size=16, embed_dim=768, num_classes = 100, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     # model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
#             map_location="cpu", check_hash=True
#         )

#         msg = model.load_state_dict(checkpoint["model"])
#         print(msg)
#     return model


# # @register_model
# def deit_tiny_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#             map_location="cpu", check_hash=True
#         )
#         head_weight_key = 'head.weight'
#         head_bias_key = 'head.bias'
#         if head_weight_key in checkpoint["model"]:
#             del checkpoint["model"][head_weight_key]
#         if head_bias_key in checkpoint["model"]:
#             del checkpoint["model"][head_bias_key]

#         msg = model.load_state_dict(checkpoint["model"], strict = False)
#         print(msg)
#     return model

# def deit_tiny_patch16_224_augsub(pretrained=False, **kwargs):
#     model = augsub_VisionTransformer(
#         num_classes=100, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     # model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#             map_location="cpu", check_hash=True
#         )
#         head_weight_key = 'head.weight'
#         head_bias_key = 'head.bias'
#         if head_weight_key in checkpoint["model"]:
#             del checkpoint["model"][head_weight_key]
#         if head_bias_key in checkpoint["model"]:
#             del checkpoint["model"][head_bias_key]

#         msg = model.load_state_dict(checkpoint["model"], strict = False)
#         print(msg)
#     return model

# def deit_small_patch16_224_aug(pretrained=False, **kwargs):
#     model = augsub_VisionTransformer(
#         num_classes=100, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model

# def vit_base_patch16(**kwargs):
#     model = augsub_VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def vit_base_patch2(**kwargs):
#     model = augsub_VisionTransformer(
#         patch_size=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, img_size = 32,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def vit_large_patch16(**kwargs):
#     model = augsub_VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = augsub_VisionTransformer(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

