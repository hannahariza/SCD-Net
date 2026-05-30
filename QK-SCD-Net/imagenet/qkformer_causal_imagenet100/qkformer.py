# from visualizer import get_local
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import functional
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import torch.nn.functional as F
from functools import partial

__all__ = ['QKFormer_8_384', 'QKFormer_10_384', 'QKFormer_10_512', 'QKFormer_10_768']


def compute_non_zero_rate(x):
    x_shape = torch.tensor(list(x.shape))
    all_neural = torch.prod(x_shape)
    z = torch.nonzero(x)
    print("After attention proj the none zero rate is", z.shape[0] / all_neural)


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
        hidden_channels = max(in_channels // 4, 1)
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )
        nn.init.constant_(self.mask_generator[-1].bias, 4.0)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_mean = x.mean(dim=0)
        mask_logits = self.mask_generator(x_mean)
        mask_prob = torch.sigmoid(mask_logits)
        mask = STEBinary.apply(mask_prob)
        x_masked = x * mask.unsqueeze(0).expand(T, B, C, H, W)
        return x_masked, mask_prob, mask


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, W, H = x.shape
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, W, H).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, W, H).contiguous()
        x = self.fc2_lif(x)
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
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

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
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.tssa(x))
        x = x + self.drop_path2(self.mlp(x))
        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = Spiking_Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(x))
        x = x + self.drop_path2(self.mlp(x))
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

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = self.proj_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
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

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
        return x


class hierarchical_spiking_transformer(nn.Module):
    def __init__(self,
                 T=4,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=256, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=10, sr_ratios=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        total_blocks = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        stage1_dpr = dpr[0:1]
        stage2_dpr = dpr[1:3]
        stage3_dpr = dpr[3:]

        self.patch_embed1 = PatchEmbedInit(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 4,
        )

        self.stage1 = nn.ModuleList([
            TokenSpikingTransformer(
                dim=embed_dims // 4,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage1_dpr[j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
            )
            for j in range(1)
        ])

        self.patch_embed2 = PatchEmbeddingStage(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 2,
        )

        self.stage2 = nn.ModuleList([
            TokenSpikingTransformer(
                dim=embed_dims // 2,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage2_dpr[j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
            )
            for j in range(2)
        ])

        self.patch_embed3 = PatchEmbeddingStage(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
        )

        self.stage3 = nn.ModuleList([
            SpikingTransformer(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage3_dpr[j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
            )
            for j in range(depths - 3)
        ])

        self.causal_mask = DynamicCausalMask(in_channels=embed_dims // 2)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed3, H, W):
        if H * W == patch_embed3.num_patches:
            return pos_embed
        return F.interpolate(
            pos_embed.reshape(1, patch_embed3.H, patch_embed3.W, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward_stage3(self, x):
        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward_features(self, x, enable_causal=True):
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)

        feat_full = self._forward_stage3(x)

        x_intervened, mask_prob, mask = self.causal_mask(x)
        functional.reset_net(self.patch_embed3)
        functional.reset_net(self.stage3)
        feat_intervened = self._forward_stage3(x_intervened)

        return feat_full, feat_intervened, mask_prob, mask

    def forward(self, x, epoch=None, enable_causal=True):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        feat_full, feat_intervened, mask_prob, mask = self.forward_features(x, enable_causal=enable_causal)

        out_full = self.head(feat_full)
        out_intervened = self.head(feat_intervened)
        firing_num_full_t = feat_full.sum(dim=2)
        firing_num_intervened_t = feat_intervened.sum(dim=2)
        pass_through_rate = mask.mean(dim=(1, 2, 3)).unsqueeze(0).expand(self.T, -1)

        return out_full, out_intervened, mask_prob, firing_num_full_t, firing_num_intervened_t, pass_through_rate


def _to_hw(img_size):
    if isinstance(img_size, int):
        return img_size, img_size
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        return int(img_size[0]), int(img_size[1])
    raise ValueError(f'Unsupported img_size: {img_size}')


def _build_qkformer(embed_dims, num_heads, depths, T=1, img_size=224, num_classes=1000,
                    in_channels=3, drop_path_rate=0., qkv_bias=False, **kwargs):
    img_h, img_w = _to_hw(img_size)
    model = hierarchical_spiking_transformer(
        T=T,
        img_size_h=img_h,
        img_size_w=img_w,
        patch_size=16,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratios=4,
        in_channels=in_channels,
        num_classes=num_classes,
        qkv_bias=qkv_bias,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=depths,
        sr_ratios=1,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    return model


def QKFormer_8_384(T=1, img_size=224, num_classes=1000, in_channels=3, drop_path_rate=0., qkv_bias=False, **kwargs):
    return _build_qkformer(384, 6, 8, T=T, img_size=img_size, num_classes=num_classes,
                           in_channels=in_channels, drop_path_rate=drop_path_rate,
                           qkv_bias=qkv_bias, **kwargs)


def QKFormer_10_384(T=1, img_size=224, num_classes=1000, in_channels=3, drop_path_rate=0., qkv_bias=False, **kwargs):
    return _build_qkformer(384, 6, 10, T=T, img_size=img_size, num_classes=num_classes,
                           in_channels=in_channels, drop_path_rate=drop_path_rate,
                           qkv_bias=qkv_bias, **kwargs)


def QKFormer_10_512(T=1, img_size=224, num_classes=1000, in_channels=3, drop_path_rate=0., qkv_bias=False, **kwargs):
    return _build_qkformer(512, 8, 10, T=T, img_size=img_size, num_classes=num_classes,
                           in_channels=in_channels, drop_path_rate=drop_path_rate,
                           qkv_bias=qkv_bias, **kwargs)


def QKFormer_10_768(T=1, img_size=224, num_classes=1000, in_channels=3, drop_path_rate=0., qkv_bias=False, **kwargs):
    return _build_qkformer(768, 12, 10, T=T, img_size=img_size, num_classes=num_classes,
                           in_channels=in_channels, drop_path_rate=drop_path_rate,
                           qkv_bias=qkv_bias, **kwargs)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224).cuda()
    model = QKFormer_10_768(T=4, img_size=224, num_classes=1000, in_channels=3).cuda()
    model.eval()
    from torchinfo import summary
    summary(model, input_size=(1, 3, 224, 224))
