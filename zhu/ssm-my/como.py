import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import GhostConv
import math

from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

_all_ = (
    "FourInputFusionBlock",
    "FusionMamba",
)

# ========== 基础模块 ==========
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm / (x.shape[-1] ** 0.5) + self.eps) * self.weight


class ECABlock(nn.Module):
    def __init__(self, ch, k_size=3):
        """
        ch: 输入通道数（用于安全检查 / 以后扩展）
        k_size: 1D 卷积核大小（must be odd）
        """
        super().__init__()
        self.ch = ch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D conv expects (B, C=1, L), we keep it as in typical ECA impl
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)                     # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)      # (B, 1, C)
        y = self.conv(y)                         # (B, 1, C)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
        return x * y.expand_as(x)


class LowRankFeedForward(nn.Module):
    
    def __init__(self, dim, rank_ratio=0.5, dropout=0.05):
        super().__init__()
        hidden_dim = max(1, int(dim * rank_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


# ========== 单Mamba块 ==========
class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W, shared_block=None, scale_init=1.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.block = shared_block or Mamba(dim, expand=1, d_state=8,
                                           bimamba_type='v6', if_devide_out=True,
                                           use_norm=True, input_h=H, input_w=W)
        self.gamma = nn.Parameter(scale_init * torch.ones(dim))
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.block(x)
        return residual + x * self.gamma


# ========== Cross Mamba ==========
class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W, rank_ratio=0.5):
        super().__init__()
        self.norm0 = RMSNorm(dim)
        self.norm1 = RMSNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8,
                           bimamba_type='v7', if_devide_out=True,
                           use_norm=True, input_h=H, input_w=W)
        self.mlp = LowRankFeedForward(dim, rank_ratio=rank_ratio)

    def forward(self, x0, x1):
        residual = x0
        x0 = self.norm0(x0)
        x1 = self.norm1(x1)
        x = self.block(x0, extra_emb=x1)
        x = x + residual
        x = x + self.mlp(x)
        return x


# ========== 主融合模块 ==========
class FusionMamba(nn.Module):
    def __init__(self, in_channels_pan, dim, H, W, depth=1, shared_weights=True):
        super().__init__()
        self.H, self.W, self.dim = H, W, dim

        # 映射层：Depthwise + Pointwise（保留轻量）
        def proj_layer(in_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.Conv2d(in_c, dim, 1, bias=False),
                nn.SiLU()
            )

        self.proj_pan = proj_layer(in_channels_pan)
        self.proj_ms  = proj_layer(in_channels_pan)

        # 动态门控池化
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.avg_pool = nn.AdaptiveAvgPool2d((H, W))
        self.max_pool = nn.AdaptiveMaxPool2d((H, W))

        shared_block = (Mamba(dim, expand=1, d_state=8,
                              bimamba_type='v6', if_devide_out=True,
                              use_norm=True, input_h=H, input_w=W)
                        if shared_weights else None)

        self.spa_layers = nn.ModuleList([SingleMambaBlock(dim, H, W, shared_block) for _ in range(depth)])
        self.spe_layers = nn.ModuleList([SingleMambaBlock(dim, H, W, shared_block) for _ in range(depth)])

        self.cross_spa = CrossMambaBlock(dim, H, W)
        self.cross_spe = CrossMambaBlock(dim, H, W)

        # 这里传入通道 dim*2（因为拼接 pan_out 与 ms_out）
        self.eca = ECABlock(dim * 2, k_size=3)
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, x):
        # device = self.proj_pan[0].weight.device
        pan, ms = x
        b, _, h0, w0 = pan.shape

        pan = self.proj_pan(pan)
        ms  = self.proj_ms(ms)

        pan_p = self.gate * self.avg_pool(pan) + (1 - self.gate) * self.max_pool(pan)
        ms_p  = self.gate * self.avg_pool(ms) + (1 - self.gate) * self.max_pool(ms)

        pan_flat = rearrange(pan_p, 'b c h w -> b (h w) c')
        ms_flat  = rearrange(ms_p,  'b c h w -> b (h w) c')
        
        for spa_layer, spe_layer in zip(self.spa_layers, self.spe_layers):
            pan_flat = spa_layer(pan_flat)
            ms_flat  = spe_layer(ms_flat)

        spa_fused = self.cross_spa(pan_flat, ms_flat)
        spe_fused = self.cross_spe(ms_flat, pan_flat)

        pan_rec = rearrange(spa_fused, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        ms_rec  = rearrange(spe_fused, 'b (h w) c -> b c h w', h=self.H, w=self.W)

        pan_out = F.interpolate(pan_rec, size=(h0, w0), mode='bilinear', align_corners=False)
        ms_out  = F.interpolate(ms_rec,  size=(h0, w0), mode='bilinear', align_corners=False)

        fused = torch.cat([pan_out, ms_out], dim=1)   # (B, dim*2, H0, W0)
        fused = self.eca(fused)                       # ECA expects ch=dim*2
        fused = self.out_conv(fused)

        return fused




def conv_bn_act(in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
    if p is None:
        p = k // 2
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class SE(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, ch, r=16):
        super().__init__()
        hidden = max(ch // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class RepBlock(nn.Module):
    """
    轻量且表现好的可重参数块（训练时：3×3 + 1×1 两支；推理时可折叠为单卷积）。
    这里保留训练结构，便于直接用；若需要折叠，再写一个reparam函数即可。
    """
    def __init__(self, ch, se=False):
        super().__init__()
        self.branch3 = conv_bn_act(ch, ch, k=3, act=True)
        self.branch1 = conv_bn_act(ch, ch, k=1, p=0, act=True)
        self.se = SE(ch) if se else nn.Identity()

    def forward(self, x):
        out = self.branch3(x) + self.branch1(x)
        out = self.se(out)
        return out + x  # 残差


class ConvBlock(nn.Module):
    """
    黄框“Conv Block”：3×3 DWConv + 1×1 PWConv（ConvNeXt风味），带SE。
    既快又稳。
    """
    def __init__(self, in_ch, out_ch, se=True):
        super().__init__()
        self.dw = conv_bn_act(in_ch, in_ch, k=3, g=in_ch, act=True)
        self.pw = conv_bn_act(in_ch, out_ch, k=1, p=0, act=True)
        self.se = SE(out_ch) if se else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.se(x)
        return x



# # --------------------------
# # FourInputFusionBlock-ultra
# # --------------------------
class GhostConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, ratio=2, act=True):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, k, s, p, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU() if act else nn.Identity()
        )
        self.cheap_op = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.primary_conv[0].out_channels * 2, :, :]


class ECABlock(nn.Module):
    def __init__(self, ch, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)

class FourInputFusionBlock(nn.Module):
    def __init__(self, in_channels=1024, use_attn=True):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * 3
        assert len(in_channels) == 3

        c = min(in_channels)   # 通道压缩到 1/4
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ic, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU()
            ) for ic in in_channels
        ])

        # 可学习融合权重（比 concat 轻量）
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.post = GhostConv(c, c, k=3, p=1)

        # 注意力
        self.attn = ECABlock(c) if use_attn else nn.Identity()

        # 输出标准化
        self.out_bn = nn.BatchNorm2d(c)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        xs = [proj(inp) for proj, inp in zip(self.proj, (x1, x2, x3))]

        # 加权融合（自适应缩放）
        w = torch.softmax(self.weights, dim=0)
        feat = sum(w[i] * xs[i] for i in range(3))

        out = self.post(feat)
        out = self.attn(out)
        out = self.out_bn(out)
        return out
# done