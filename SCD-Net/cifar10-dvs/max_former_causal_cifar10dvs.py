import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based import layer, neuron
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from mixer_hub import *
from embedding_hub import *
from spikingjelly.clock_driven import functional

__all__ = ['max_former_causal']


class STEBinary(torch.autograd.Function):
    """
    直通估计器 (Straight-Through Estimator)
    前向传播：输出 0 或 1
    反向传播：将梯度直接传回
    """

    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DynamicCausalMask(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 使用 SpikingJelly 的 layer 并设置 step_mode='m'
        # 这使得 Sequential 可以直接吞吐 [T, B, C, H, W] 的五维张量
        self.mask_generator = nn.Sequential(
            layer.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(in_channels // 4, step_mode='m'),
            neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m'),
            layer.Conv2d(in_channels // 4, in_channels, kernel_size=1, bias=False, step_mode='m')
        )

    def forward(self, x):
        # x shape 始终为: [T, B, C, H, W]
        T, B, C, H, W = x.shape

        # 1. 直接处理五维张量
        # SNN 的时序特征（在特定 t 时刻的脉冲发放）被完美保留和提取
        # mask_logits 形状自动为 [T, B, C, H, W]
        mask_logits = self.mask_generator(x)

        # 2. 生成基于时序脉冲的动态保留概率
        prob = torch.sigmoid(mask_logits)  # [T, B, C, H, W]

        # 3. STE 二值化 (前向变为绝对的 0/1 矩阵，反向透传梯度)
        mask = STEBinary.apply(prob)  # [T, B, C, H, W]

        # 4. 纯物理按位与 (AND) 干预
        out_masked = x * mask

        return out_masked, prob


class Max_Former(nn.Module):
    def __init__(self,
                 in_channels=2, num_classes=10,
                 embed_dims=[64, 128, 256], mlp_ratios=[4, 4, 4],
                 depths=[6, 8, 6], T=4
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        patch_embed1 = Embed_Max_plus(in_channels=in_channels,
                                      embed_dims=embed_dims // 2)

        stage1 = nn.ModuleList(
            [Block_DWC3(
                dim=embed_dims // 2, mlp_ratio=mlp_ratios)
                for j in range(1)]
        )

        patch_embed2 = Embed_Max(in_channels=embed_dims // 2,
                                 embed_dims=embed_dims)

        stage2 = nn.ModuleList([Block_SSA(
            dim=embed_dims, mlp_ratio=mlp_ratios, num_heads=16)
            for j in range(1)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)

        # === 插入 Causal Mask ===
        self.causal_mask = DynamicCausalMask(in_channels=embed_dims // 2)

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

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

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        # === Causal Mask 干预及拼接并行运算 ===
        x_full = x
        x_masked, mask_prob = self.causal_mask(x_full)

        x_full_deep = patch_embed2(x_full)
        for blk in stage2:
            x_full_deep = blk(x_full_deep)

        functional.reset_net(patch_embed2)
        functional.reset_net(stage2)

        x_masked_deep = patch_embed2(x_masked)
        for blk in stage2:
            x_masked_deep = blk(x_masked_deep)

            # 计算深层特征时序放电统计信息
        # T, B, C_out, H_out, W_out = x_masked_deep.shape
        # firing_num_t = x_masked_deep.sum(dim=(2, 3, 4))  # [T, B]
        # firing_rate = firing_num_t / (C_out * H_out * W_out)

        feat_full = x_full_deep.flatten(3).mean(3)
        feat_masked = x_masked_deep.flatten(3).mean(3)

        #return feat_full, feat_masked, mask_prob, firing_num_t, firing_rate

        return feat_full, feat_masked, mask_prob

    def forward(self, x):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        # 获取特征表示及因果干预统计量
        #x_full, x_masked, mask_prob, firing_num_t, firing_rate = self.forward_features(x)
        x_full, x_masked, mask_prob = self.forward_features(x)

        feat_full_lif = self.head_lif(x_full)
        firing_num_full_t = feat_full_lif.sum(dim=2)  # [T, B]
        #out_full = self.head(feat_full_lif.mean(0))
        out_full = self.head(feat_full_lif)

        functional.reset_net(self.head_lif)  # 重置 LIF 状态以处理第二路特征

        feat_masked_lif = self.head_lif(x_masked)
        firing_num_masked_t = feat_masked_lif.sum(dim=2)  # [T, B]
        firing_rate = feat_masked_lif.mean(dim=2)  # [T, B]
        #out_masked = self.head(feat_masked_lif.mean(0))
        out_masked = self.head(feat_masked_lif)

        functional.reset_net(self.head_lif)

        #return out_full, out_masked, mask_prob, firing_num_t, firing_rate

        return out_full, out_masked, mask_prob, firing_num_full_t, firing_num_masked_t, firing_rate


@register_model
def max_former_causal(pretrained=False, pretrained_cfg=None, **kwargs):
    model = Max_Former(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    # 测试实例化及参数量
    model = max_former_causal(
        embed_dims=256, mlp_ratios=1.0,
        in_channels=2, num_classes=10, depths=2
    ).cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {num_params}")

    # 构建模拟输入 [B, T, C, H, W]
    # 注意，这里的网络 forward 内部做了 .transpose(0, 1) 转换
    input = torch.randn(2, 4, 2, 128, 128).cuda()

    # 推理测试
    out_full, out_masked, mask_prob, firing_num_t, firing_rate = model(input)

    print("\n[ Max_Former Causal 模型前向传播测试通过 ]")
    print(f"out_full shape: {out_full.shape}")  # 期望: [2, 10]
    print(f"out_masked shape: {out_masked.shape}")  # 期望: [2, 10]
    print(f"mask_prob shape: {mask_prob.shape}")  # 期望: [4, 2, 128, H_stage1, W_stage1]
    print(f"firing_num_t shape: {firing_num_t.shape}")  # 期望: [4, 2]
    print(f"firing_rate shape: {firing_rate.shape}")  # 期望: [4, 2]