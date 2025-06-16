
from .base_model import BaseModel
from collections import OrderedDict


import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

import math
import torch.nn.functional as F
from einops import repeat
from typing import Sequence, Type, Optional
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass


class HWTB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),   
            nn.ReLU(inplace=False),                                 
        ) 

    def forward(self, M):
        M_L, M_H = self.wt(M)
        M_HH = M_H[0][:,:,0,::]
        M_HV = M_H[0][:,:,1,::]
        M_HD = M_H[0][:,:,2,::]
        M_ = torch.cat([M_L, M_HH, M_HV, M_HD], dim=1)        
        x = self.layer(M_)
        return x

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)
    
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x    

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        # x tranform form (b, d, h, w)
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        try:
            y = self.out_norm(y)
        except:
            y = self.out_norm.to(torch.float32)(y).half()

        y = y * F.silu(z)
        try:
            out = self.out_proj(y)
        except:
            out = self.out_proj.to(torch.float32)(y).half()
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out


class ChannelAttention(nn.Module):
    def __init__(self, out_features: int, ratio: int = 8) -> None:
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(out_features, out_features // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features // ratio, out_features, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale  

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class DA_FFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
    ) -> None:
        super(DA_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features*4, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        # depthwise convolution
        self.dwconv_1x1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0, groups=hidden_features)
        self.dwconv_3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.dwconv_5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding=2, groups=hidden_features)

        self.fusion = nn.Conv2d(hidden_features * 4, out_features, kernel_size=1)
        self.ca = ChannelAttention(out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # 
        x_chunks = torch.chunk(x, 4, dim=1)

        x0 = x_chunks[0]         # residual 
        x1 = self.dwconv_1x1(x_chunks[1])
        x2 = self.dwconv_3x3(x_chunks[2])
        x3 = self.dwconv_5x5(x_chunks[3])

        # concat
        x = torch.cat([x0, x1, x2, x3], dim=1)   # shape: [B, 4C, H, W]
        x = self.fusion(x)                      # shape: [B, C, H, W]

        x = self.ca(x)                          

        return x

class CVSSB(nn.Module):
    """ 
    [B,in_dim,H,W] -> [B,out_dim,H,W]
    """
    def __init__(
        self,
        in_dim,
        out_dim
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm2d(in_dim)
        self.relu = nn.ReLU()
        self.ss2d = SS2D(d_model=in_dim)

        self.res_proj = (
            nn.Identity() if out_dim == in_dim else nn.Conv2d(in_dim, out_dim, kernel_size=1)
        )

        self.ln2 = LayerNorm2d(in_dim)
        self.ffn = DA_FFN(
            in_features=in_dim,
            hidden_features=in_dim,
            out_features=out_dim
        )

    def forward(
            self,
            x
    ):
        x_ = x

        x = self.ln1(x)
        x = self.ss2d(x)
        x = x + x_

        x_ = self.res_proj(x)

        x = self.ln2(x)
        x = self.ffn(x)

        x = x + x_

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        return x
    

class PatchExpansion(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super(PatchExpansion, self).__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 128→256
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),  # 32→16
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 256→512
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),  # 16→8
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)  # [B, 16, 256, 256]
        x = self.up2(x)  # [B, 8, 512, 512]
        x = self.out_conv(x)  # [B, 1, 512, 512]
        return x
    

class EAGLE(BaseModel):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        patch_size=4,               # stem h/4, w/4
        c_list=[32,64,128,256,512],
        layer_num=[2,2,4,2],
    ):
        super(EAGLE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.c_list = c_list

        # ==================== PatchEmbedding ==================== # 
        self.stem = PatchEmbedding(in_channels=in_channels, embed_dim=c_list[0], patch_size=patch_size)

        # ==================== Encoder ==================== # 
        # ----- encoder block 1 ----- #
        cvssb1_blocks = []
        cvssb1_blocks.append(CVSSB(c_list[0], c_list[1]))
        for _ in range(layer_num[0]-1):
            cvssb1_blocks.append(CVSSB(c_list[1], c_list[1]))
        cvssb1_blocks.append(HWTB(c_list[1], c_list[1]))
        self.encoder1 = nn.Sequential(*cvssb1_blocks)

        # ----- encoder block 2 ----- #
        cvssb2_blocks = []
        cvssb2_blocks.append(CVSSB(c_list[1], c_list[2]))
        for _ in range(layer_num[1]-1):
            cvssb2_blocks.append(CVSSB(c_list[2], c_list[2]))
        cvssb2_blocks.append(HWTB(c_list[2], c_list[2]))
        self.encoder2 = nn.Sequential(*cvssb2_blocks)

        # ----- encoder block 3 ----- #
        cvssb3_blocks = []
        cvssb3_blocks.append(CVSSB(c_list[2], c_list[3]))
        for _ in range(layer_num[2]-1):
            cvssb3_blocks.append(CVSSB(c_list[3], c_list[3]))
        cvssb3_blocks.append(CBAMLayer(c_list[3]))
        cvssb3_blocks.append(HWTB(c_list[3], c_list[3]))
        self.encoder3 = nn.Sequential(*cvssb3_blocks)

        # ----- encoder block 4 ----- #
        cvssb4_blocks = []
        cvssb4_blocks.append(CVSSB(c_list[3], c_list[4]))
        for _ in range(layer_num[3]-1):
            cvssb4_blocks.append(CVSSB(c_list[4], c_list[4]))
        cvssb4_blocks.append(CBAMLayer(c_list[4]))
        cvssb4_blocks.append(HWTB(c_list[4], c_list[4]))
        self.encoder4 = nn.Sequential(*cvssb4_blocks)
        # ==================== Encoder END ==================== # 

        # ==================== Decoder ==================== # 
        # ----- decoder block 4 ----- #
        cvssb4_blocks = []
        cvssb4_blocks.append(CVSSB(c_list[4], c_list[3]))
        for _ in range(layer_num[3]-1):
            cvssb4_blocks.append(CVSSB(c_list[3], c_list[3]))
        cvssb4_blocks.append(CBAMLayer(c_list[3]))
        cvssb4_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder4 = nn.Sequential(*cvssb4_blocks)

        # ----- decoder block 3 ----- #
        cvssb3_blocks = []
        cvssb3_blocks.append(CVSSB(c_list[3], c_list[2]))
        for _ in range(layer_num[2]-1):
            cvssb3_blocks.append(CVSSB(c_list[2], c_list[2]))
        cvssb3_blocks.append(CBAMLayer(c_list[2]))
        cvssb3_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder3 = nn.Sequential(*cvssb3_blocks)

        # ----- decoder block 2 ----- #
        cvssb2_blocks = []
        cvssb2_blocks.append(CVSSB(c_list[2], c_list[1]))
        for _ in range(layer_num[1]-1):
            cvssb2_blocks.append(CVSSB(c_list[1], c_list[1]))
        cvssb2_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder2 = nn.Sequential(*cvssb2_blocks)

        # ----- decoder block 1 ----- #
        cvssb1_blocks = []
        cvssb1_blocks.append(CVSSB(c_list[1], c_list[0]))
        for _ in range(layer_num[0]-1):
            cvssb1_blocks.append(CVSSB(c_list[0], c_list[0]))
        cvssb1_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder1 = nn.Sequential(*cvssb1_blocks)

        # ==================== Decoder END ==================== # 

        # ==================== Patch Expansion ====================# 
        self.patch_exp = PatchExpansion(in_channels=c_list[0], out_channels=out_channels)
        # ==================== Patch Expansion END =========# 

    
    def forward(self, x):

        stem_out = self.stem(x)  # [B, 32, 128, 128]

        e1 = self.encoder1(stem_out)   # [B, 64, 64, 64]
        e2 = self.encoder2(e1)              # [B, 128, 32, 32]
        e3 = self.encoder3(e2)              # [B, 256, 16, 16]
        e4 = self.encoder4(e3)              # [B, 512, 8, 8]

        d4 = self.decoder4(e4) + e3         # [B, 256, 16, 16]
        d3 = self.decoder3(d4) + e2         # [B, 128, 32, 32]
        d2 = self.decoder2(d3) + e1         # [B, 64, 64, 64]
        d1 = self.decoder1(d2) + stem_out   # [B, 32, 128, 128]

        out = self.patch_exp(d1)            # [B, 1, 512, 512]

        return torch.sigmoid(out)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EAGLE(
        in_channels=1,
        out_channels=1,
        patch_size=4,
        c_list=[32,64,128,256,512],
        layer_num=[2,2,4,2],
    ).to(device)   

    x = torch.randn(1, 1, 512, 512).to(device)  

    y = model(x)
    print(y.shape)