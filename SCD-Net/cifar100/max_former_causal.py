import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mixer_hub import *
from embedding_hub import *
import torch.nn.functional as F
from spikingjelly.clock_driven import functional

__all__ = ['max_former']


class STEBinary(torch.autograd.Function):
    """
    直通估计器 (Straight-Through Estimator)
    前向传播：输出 0 或 1
    反向传播：将梯度直接传回
    """

    @staticmethod
    def forward(ctx, input):
        # 这里的阈值可以设为 0.5 或者根据需要调整
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 直接透传梯度
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

    def forward(self, x):
        # x shape: [T, B, C, H, W]
        T, B, C, H, W = x.shape


        x_mean = x.mean(dim=0)  # [B, C, H, W]

        mask_logits = self.mask_generator(x_mean)
        prob = torch.sigmoid(mask_logits)  # [B, C, H, W]

        mask = STEBinary.apply(prob)  # [B, C, H, W]

        mask = mask.unsqueeze(0).expand(T, B, C, H, W)


        out_masked = x * mask

        return out_masked, prob


class Max_Former(nn.Module):
    def __init__(self, in_channels=2, num_classes=11,
                 embed_dims=384, mlp_ratios=4, drop_rate=0.,
                 depths=[6, 8, 6], T=4
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        patch_embed1 = Embed_Orig(in_channels=in_channels,
                                  embed_dims=embed_dims // 4)

        stage1 = nn.ModuleList([Block_identity(
            dim=embed_dims // 4, mlp_ratio=mlp_ratios)
            for j in range(1)])

        patch_embed2 = Embed_Max(in_channels=embed_dims // 4,
                                 embed_dims=embed_dims // 2)

        stage2 = nn.ModuleList([Block_DWC3(
            dim=embed_dims // 2, mlp_ratio=mlp_ratios, )
            for j in range(1)])

        patch_embed3 = Embed_Max(in_channels=embed_dims // 2,
                                 embed_dims=embed_dims // 1)

        stage3 = nn.ModuleList([Block_SSA(
            dim=embed_dims // 1, mlp_ratio=mlp_ratios,
            num_heads=8, )
            for j in range(2)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        self.causal_mask = DynamicCausalMask(in_channels=embed_dims // 2)

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        for blk in stage1: x = blk(x)
        x = patch_embed2(x)
        for blk in stage2: x = blk(x)

        x_full = x
        x_masked, mask_prob = self.causal_mask(x_full)

        x_full_deep = patch_embed3(x_full)
        for blk in stage3:
            x_full_deep = blk(x_full_deep)


        functional.reset_net(patch_embed3)
        functional.reset_net(stage3)

        x_masked_deep = patch_embed3(x_masked)
        for blk in stage3:
            x_masked_deep = blk(x_masked_deep)


        feat_full = x_full_deep.flatten(3).mean(3)
        feat_masked = x_masked_deep.flatten(3).mean(3)


        return feat_full, feat_masked, mask_prob

    def forward(self, x, epoch=None):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        feat_full, feat_masked, mask_prob = self.forward_features(x)

        feat_full_lif = self.head_lif(feat_full)  # [T, B, C]
        firing_num_full_t = feat_full_lif.sum(dim=2)  # [T, B]
        out_full = self.head(feat_full_lif)  # [T, B, num_classes]

        functional.reset_net(self.head_lif)

        feat_masked_lif = self.head_lif(feat_masked)
        firing_num_masked_t = feat_masked_lif.sum(dim=2)  # [T, B]
        firing_rate = feat_masked_lif.mean(dim=2)  # [T, B]
        out_masked = self.head(feat_masked_lif)  # [T, B, num_classes]

        functional.reset_net(self.head_lif)


        return out_full, out_masked, mask_prob, firing_num_full_t, firing_num_masked_t, firing_rate


@register_model
def max_former(pretrained=False, pretrained_cfg=None, **kwargs):
    model = Max_Former(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    model = Max_Former(
        embed_dims=384, mlp_ratios=4,
        in_channels=3, num_classes=10, depths=4
    ).cuda()

    input = torch.randn(4, 3, 32, 32).cuda()


    out_full, out_masked, mask_prob, firing_num_full_t, firing_num_masked_t, firing_rate = model(input)

    print("out_full shape:", out_full.shape)
    print("out_masked shape:", out_masked.shape)
    print("mask_prob shape:", mask_prob.shape)
    print("firing_num_full_t shape:", firing_num_full_t.shape)
    print("firing_num_masked_t shape:", firing_num_masked_t.shape)
    print("firing_rate shape:", firing_rate.shape)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")