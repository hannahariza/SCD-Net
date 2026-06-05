import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
from spikingjelly.activation_based import layer, neuron
from spikingjelly.clock_driven import functional
import torch.nn.functional as F

__all__ = ['QKFormer']


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
        # 在神经形态硬件上，这体现为极低能耗的脉冲路由开关 (Routing Gating)
        out_masked = x * mask

        # # 5. 计算干预后的放电信息
        # firing_num_t = out_masked.sum(dim=(2, 3, 4))  # [T, B]
        # firing_rate = firing_num_t / (C * H * W)

        #return out_masked, prob, firing_num_t, firing_rate

        return out_masked, prob


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # depthwise 3x3 卷积 (新增)
        self.dw_conv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            padding=1,
            groups=hidden_features,
            bias=False
        )
        self.dw_bn = nn.BatchNorm2d(hidden_features)
        self.dw_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)

        # 3x3 depthwise
        x = self.dw_conv(x.flatten(0, 1))
        x = self.dw_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.dw_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x = self.mlp2_lif(x)
        return x


class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)
        return x


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)
        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = Spiking_Self_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.ssa(x)
        x = x + self.mlp(x)
        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.maxpool1(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.maxpool2(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.maxpool3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj3_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        x = x + x_feat  # shortcut
        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut
        return x


class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None, in_chans=3, no_weight_decay=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        num_heads = [16, 16, 16]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                      img_size_w=img_size_w,
                                      patch_size=patch_size,
                                      in_channels=in_channels,
                                      embed_dims=embed_dims // 2)

        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                           img_size_w=img_size_w,
                                           patch_size=patch_size,
                                           in_channels=in_channels,
                                           embed_dims=embed_dims)

        stage2 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads[1], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"stage1", stage1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage2", stage2)

        # 在 stage1 和 stage2 之间添加 Causal Mask (因为 DVS 基础网络共2个stage)
        self.causal_mask = DynamicCausalMask(in_channels=embed_dims // 2)

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

    def forward_features(self, x, epoch=0):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        # === 插入 Causal Mask ===
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

        return feat_full, feat_masked, mask_prob

        # # 将原始特征和干预后的特征拼接，使得后续网络一次并行计算
        # x_concat = torch.cat([x_full, x_masked], dim=1)
        #
        # x_concat = patch_embed2(x_concat)
        # for blk in stage2:
        #     x_concat = blk(x_concat)
        #
        # # 解开拆分回 full 和 masked
        # x_full_deep, x_masked_deep = torch.chunk(x_concat, 2, dim=1)
        #
        # # 记录深层特征在各个时间步的发放总数与平均放电率
        # T, B, C_out, H_out, W_out = x_masked_deep.shape
        # firing_num_t = x_masked_deep.sum(dim=(2, 3, 4))  # [T, B]
        # firing_rate = firing_num_t / (C_out * H_out * W_out)
        #
        # feat_full = x_full_deep.flatten(3).mean(3)
        # feat_masked = x_masked_deep.flatten(3).mean(3)
        #
        # return feat_full, feat_masked, mask_prob, firing_num_t, firing_rate

    def forward(self, x, epoch=0):
        # DVS 数据的常见排列: [B, T, C, H, W] -> 传给 forward_features 时为 [T, B, C, H, W]
        x = x.permute(1, 0, 2, 3, 4)
        # x_full, x_masked, mask_prob, firing_num_t, firing_rate = self.forward_features(x, epoch)
        x_full, x_masked, mask_prob = self.forward_features(x)

        firing_num_full_t = x_full.sum(dim=2)  # [T, B]
        out_full = self.head(x_full)

        firing_num_masked_t = x_masked.sum(dim=2)  # [T, B]
        firing_rate = x_masked.mean(dim=2)  # [T, B]
        out_masked = self.head(x_masked)

        # out_full = self.head(x_full.mean(0))
        # out_masked = self.head(x_masked.mean(0))

        return out_full, out_masked, mask_prob, firing_num_full_t, firing_num_masked_t, firing_rate


@register_model
def QKFormer(pretrained=False, **kwargs):
    model = vit_snn(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=1,
        in_channels=2, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    # CIFAR10-DVS 的默认输入尺寸测试: [BatchSize, Time, Channels, Height, Width]
    # 使用 Batch Size = 2，T = 4 来验证拼接、拆解以及时序维度的处理是否正确
    x = torch.randn(2, 4, 2, 128, 128).cuda()

    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
    ).cuda()
    model.eval()

    from torchinfo import summary

    # 在 torchinfo 摘要里显示网络结构概况
    summary(model, input_size=(2, 4, 2, 128, 128), epoch=0)

    out_full, out_masked, mask_prob, firing_num_t, firing_rate = model(x, epoch=0)
    print("\n[ 测试通过 ]")
    print(f"out_full shape: {out_full.shape}")
    print(f"out_masked shape: {out_masked.shape}")
    print(f"mask_prob shape: {mask_prob.shape}")
    print(f"firing_num_t shape: {firing_num_t.shape}")
