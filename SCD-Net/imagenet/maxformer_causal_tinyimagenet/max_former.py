import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
)
from model.maxformer_causal_tiny_imagenet.mixer_hub import *
from model.maxformer_causal_tiny_imagenet.embedding_hub import *
import torch.nn.functional as F
from spikingjelly.clock_driven import functional

__all__ = ['max_former']

class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class DynamicCausalMask(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        )
        nn.init.constant_(self.mask_generator[-1].bias, 4.0)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_mean = x.mean(dim=0)
        mask_logits = self.mask_generator(x_mean)
        prob = torch.sigmoid(mask_logits)
        mask = STEBinary.apply(prob)
        mask = mask.unsqueeze(0).expand(T, B, C, H, W)
        out_masked = x * mask
        return out_masked, prob

class Max_Former(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        mlp_ratios=4,
        depths=[6, 8, 6],
        T=4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        patch_embed1 = Embed_Orig_ImageNet(
            in_channels=in_channels,
            embed_dims=embed_dims // 4,
        )

        stage1 = nn.ModuleList(
            [  Block_DWC7( 
                    dim=embed_dims // 4, mlp_ratio=mlp_ratios,
                    ) for j in range(1)
            ]
        )

        patch_embed2 =Embed_Max(
            in_channels=embed_dims // 4,
            embed_dims=embed_dims // 2,
        )

        stage2 = nn.ModuleList(
            [   Block_DWC5(
                    dim=embed_dims // 2, mlp_ratio=mlp_ratios,
                    ) for j in range(2)
            ]
        )

        patch_embed3 = Embed_Max(
            in_channels=embed_dims // 2,
            embed_dims=embed_dims // 1,
        )

        stage3 = nn.ModuleList(
            [   Block_SSA(
                    dim=embed_dims // 1, mlp_ratio=mlp_ratios, num_heads = embed_dims // 64,
                   ) for j in range(7)
            ]
        )

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        self.causal_mask = DynamicCausalMask(in_channels=embed_dims // 2)

        # classification head
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.head = (nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, enable_causal=True):
        stage1 = getattr(self, f"stage1")
        stage2 = getattr(self, f"stage2")
        stage3 = getattr(self, f"stage3")

        patch_embed1 = getattr(self, f"patch_embed1")
        patch_embed2 = getattr(self, f"patch_embed2")
        patch_embed3 = getattr(self, f"patch_embed3")


        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x_full_deep = patch_embed3(x)
        for blk in stage3:
            x_full_deep = blk(x_full_deep)
        feat_full = x_full_deep.flatten(3).mean(3)

        if not enable_causal:
            return feat_full, None, None

        x_masked, mask_prob = self.causal_mask(x)

        functional.reset_net(patch_embed3)
        functional.reset_net(stage3)

        x_masked_deep = patch_embed3(x_masked)
        for blk in stage3:
            x_masked_deep = blk(x_masked_deep)
        feat_masked = x_masked_deep.flatten(3).mean(3)

        return feat_full, feat_masked, mask_prob

    def forward(self, x, epoch=None, enable_causal=True):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        feat_full, feat_masked, mask_prob = self.forward_features(x, enable_causal=enable_causal)

        # 完整特征分支 (保留 T 维度)
        feat_full_lif = self.head_lif(feat_full)  # [T, B, C]
        firing_num_full_t = feat_full_lif.sum(dim=2)  # [T, B]
        out_full = self.head(feat_full_lif)  # [T, B, num_classes]

        functional.reset_net(self.head_lif)

        if not enable_causal:
            return out_full, None, None, firing_num_full_t, None, None

        # 掩码干预分支 (保留 T 维度)
        feat_masked_lif = self.head_lif(feat_masked)
        firing_num_masked_t = feat_masked_lif.sum(dim=2)  # [T, B]
        firing_rate = feat_masked_lif.mean(dim=2)  # [T, B]
        out_masked = self.head(feat_masked_lif)  # [T, B, num_classes]

        functional.reset_net(self.head_lif)

        # 返回 6 个值，包含两个分支的发放计数
        return out_full, out_masked, mask_prob, firing_num_full_t, firing_num_masked_t, firing_rate
    
@register_model
def maxformer(pretrained = False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Max_Former(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def maxformer_10_384(pretrained = False, pretrained_cfg=None, pretrained_cfg_overlay=None, T=1, **kwargs):
    num_classes = kwargs.pop('num_classes', 100)
    model = Max_Former(
        T=T, embed_dims=384, mlp_ratios=4,
        in_channels=3, num_classes=num_classes,
        depths=10, 
        **kwargs
    )
    return model

@register_model
def maxformer_10_512(pretrained = False, pretrained_cfg=None, pretrained_cfg_overlay=None, T=1, **kwargs):
    num_classes = kwargs.pop('num_classes', 100)
    model = Max_Former(
        T=T, embed_dims=512, mlp_ratios=4,
        in_channels=3, num_classes=num_classes,
        depths=10, 
        **kwargs
    )
    return model

@register_model
def maxformer_10_768(pretrained = False, pretrained_cfg=None, pretrained_cfg_overlay=None, T=1, **kwargs):
    num_classes = kwargs.pop('num_classes', 100)
    model = Max_Former(
        T=T, embed_dims=768, mlp_ratios=4,
        in_channels=3, num_classes=num_classes,
        depths=10, 
        **kwargs
    )
    return model