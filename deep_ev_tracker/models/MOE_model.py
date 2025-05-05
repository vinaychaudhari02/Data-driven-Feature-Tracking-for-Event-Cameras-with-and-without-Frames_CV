"""
An end-to-end tracking module that preserves the original FPN+ConvLSTM spatial backbone
and augments it with MvHeat_DET frequency-domain features, all in one file with no external model imports.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Callable
from functools import partial
from models.template import Template
from PIL import Image as _PilImage
if not hasattr(_PilImage, "ANTIALIAS"):
    _PilImage.ANTIALIAS = _PilImage.LANCZOS
#from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

# --- Registry stub ---
def register(cls):
    """No-op decorator for model registration"""
    return cls


# --- Loss stub ---
class L1Truncated(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = F.l1_loss(pred, target)
        # Return scalar mask (count of items) for aggregation
        mask = torch.tensor(1.0, device=pred.device)
        return loss, mask

# --- LayerScale ---
class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

# --- ConvLSTMCell from user code ---
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.Gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.prev_state = None
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.Gates.weight)

    def reset(self):
        self.prev_state = None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        b, _, h, w = input_.size()
        if self.prev_state is None:
            device = input_.device
            state_size = [b, self.hidden_channels, h, w]
            self.prev_state = (
                torch.zeros(state_size, device=device),
                torch.zeros(state_size, device=device),
            )
        prev_h, prev_c = self.prev_state
        stacked = torch.cat([input_, prev_h], dim=1)
        gates = self.Gates(stacked)
        i, r, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        r = torch.sigmoid(r)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = r * prev_c + i * g
        h_new = o * torch.tanh(c)
        self.prev_state = (h_new, c)
        return h_new

# --- ConvBlock from user code ---
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_convs: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = True,
        dilation: int = 1,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(
                    c_in, out_channels,
                    kernel_size, stride, padding,
                    dilation=dilation, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            c_in = out_channels
        if downsample:
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(),
            ]
        self.model = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- FPNEncoder from user code ---
class FPNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 512, recurrent: bool = False):
        super().__init__()
        self.conv_bottom_0 = ConvBlock(in_channels, 32, n_convs=2, kernel_size=1, padding=0, downsample=False)
        self.conv_bottom_1 = ConvBlock(32, 64, n_convs=2, kernel_size=5, padding=0, downsample=False)
        self.conv_bottom_2 = ConvBlock(64, 128, n_convs=2, kernel_size=5, padding=0, downsample=False)
        self.conv_bottom_3 = ConvBlock(128, 256, n_convs=2, kernel_size=3, padding=0, downsample=True)
        self.conv_bottom_4 = ConvBlock(256, out_channels, n_convs=2, kernel_size=3, padding=0, downsample=False)
        self.recurrent = recurrent
        if recurrent:
            self.conv_rnn = ConvLSTMCell(out_channels, out_channels, 1)

        # lateral & dealias layers
        self.conv_lateral_3 = nn.Conv2d(256, out_channels, kernel_size=1, bias=True)
        self.conv_lateral_2 = nn.Conv2d(128, out_channels, kernel_size=1, bias=True)
        self.conv_lateral_1 = nn.Conv2d(64, out_channels, kernel_size=1, bias=True)
        self.conv_lateral_0 = nn.Conv2d(32, out_channels, kernel_size=1, bias=True)
        self.conv_dealias_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dealias_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dealias_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dealias_0 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv_out = nn.Sequential(
            ConvBlock(out_channels, out_channels, n_convs=1, kernel_size=3, padding=1, downsample=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )
        self.conv_bottleneck_out = nn.Sequential(
            ConvBlock(out_channels, out_channels, n_convs=1, kernel_size=3, padding=1, downsample=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def reset(self):
        if self.recurrent:
            self.conv_rnn.reset()

    def forward(self, x: torch.Tensor):
        c0 = self.conv_bottom_0(x)
        c1 = self.conv_bottom_1(c0)
        c2 = self.conv_bottom_2(c1)
        c3 = self.conv_bottom_3(c2)
        c4 = self.conv_bottom_4(c3)
        p4 = c4
        p3 = self.conv_dealias_3(
            self.conv_lateral_3(c3)
            + F.interpolate(p4, size=c3.shape[2:], mode="bilinear")
        )
        p2 = self.conv_dealias_2(
            self.conv_lateral_2(c2)
            + F.interpolate(p3, size=c2.shape[2:], mode="bilinear")
        )
        p1 = self.conv_dealias_1(
            self.conv_lateral_1(c1)
            + F.interpolate(p2, size=c1.shape[2:], mode="bilinear")
        )
        p0 = self.conv_dealias_0(
            self.conv_lateral_0(c0)
            + F.interpolate(p1, size=c0.shape[2:], mode="bilinear")
        )
        if self.recurrent:
            p0 = self.conv_rnn(p0)
        return self.conv_out(p0), self.conv_bottleneck_out(c4)

# --- MvHeat_DET and helpers from user code ---
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()

class to_channels_first(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()

class to_channels_last(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()
    
def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')
    
class StemLayer(nn.Module):
    def __init__(self, in_chans=3, out_chans=96, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first', 'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 3)
        )

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return hard_mask

class Heat2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.policy = PolicyNet(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def haar_transform_1d(self, x):
        n = x.shape[-1]
        assert n % 2 == 0, "Input length must be even for Haar transform"
        avg = (x[..., ::2] + x[..., 1::2]) / 2
        diff = (x[..., ::2] - x[..., 1::2]) / 2
        return torch.cat((avg, diff), dim=-1)
    
    def haar_transform(self, x, dims=(-2, -1)):
        transformed = x.clone()
        for dim in dims:
            shape = transformed.shape
            transformed = transformed.view(-1, shape[dim])
            transformed = self.haar_transform_1d(transformed)
            transformed = transformed.view(*shape)
        return transformed
    
    def inverse_haar_transform_1d(self, x):
        n = x.shape[-1]
        assert n % 2 == 0, "Input length must be even for inverse Haar transform"
        avg = x[..., :n // 2]
        diff = x[..., n // 2:]
        x_rec = torch.zeros(x.shape[:-1] + (n,), device=x.device)
        x_rec[..., ::2] = avg + diff
        x_rec[..., 1::2] = avg - diff
        return x_rec
    
    def inverse_haar_transform(self, x, dims=(-2, -1)):
        transformed = x.clone()
        for dim in dims:
            shape = transformed.shape
            transformed = transformed.view(-1, shape[dim])
            transformed = self.inverse_haar_transform_1d(transformed)
            transformed = transformed.view(*shape)
        return transformed

    def forward(self, x: torch.Tensor, freq_embed=None):
        # Record original shape and pad to even H/W if needed
        B, C, H, W = x.shape
        orig_H, orig_W = H, W
        pad_bottom = H % 2
        pad_right = W % 2
        if pad_bottom or pad_right:
            x = F.pad(x, (0, pad_right, 0, pad_bottom))
        # Now continue as before, but use possibly padded H/W
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(2, dim=-1)

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__")
            weight_cosm = getattr(self, "__WEIGHT_COSM__")
            weight_exp = getattr(self, "__WEIGHT_EXP__")
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        moe = self.policy(x.mean(dim=(1,2)), 1)

        x_dct = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(weight_cosn.size(0), H, 1))
        x_dct = F.conv1d(x_dct.contiguous().view(-1, W, C), weight_cosm.contiguous().view(weight_cosm.size(0), W, 1))
        x_dct = x_dct.view(B, weight_cosn.size(0), weight_cosm.size(0), -1)

        x_fft = torch.fft.fftn(x)
        x_haar = self.haar_transform(x)

        if self.infer_mode:
            x_dct = torch.einsum("bnmc,nmc->bnmc", x_dct, self.k_exp)
            x_fft = torch.einsum("bnmc,nmc->bnmc", x_fft, self.k_exp)
            x_haar = torch.einsum("bnmc,nmc->bnmc", x_haar, self.k_exp)
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x_dct = torch.einsum("bnmc,nmc->bnmc", x_dct, weight_exp)
            x_fft = torch.einsum("bnmc,nmc->bnmc", x_fft, weight_exp)
            x_haar = torch.einsum("bnmc,nmc->bnmc", x_haar, weight_exp)

        x_dct = F.conv1d(x_dct.contiguous().view(B, weight_cosn.size(0), -1), weight_cosn.t().contiguous().view(H, weight_cosn.size(0), 1))
        x_dct = F.conv1d(x_dct.contiguous().view(-1, weight_cosm.size(0), C), weight_cosm.t().contiguous().view(W, weight_cosm.size(0), 1))
        x_dct = x_dct.view(B, H, W, -1)

        x_fft = torch.fft.ifftn(x_fft).real
        x_haar = self.inverse_haar_transform(x_haar)

        x = torch.cat((x_dct.unsqueeze(1), x_fft.unsqueeze(1), x_haar.unsqueeze(1)), dim=1)
        x = torch.einsum("brnmc,br->bnmc", x, moe)
        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_linear(x)
        # Permute to (B, C, N, M) and crop back to original shape if padded
        x = x.permute(0, 3, 1, 2).contiguous()
        # Crop padded dimensions back to original size
        if pad_bottom or pad_right:
            x = x[..., :orig_H, :orig_W].contiguous()
        return x

class HeatBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode=False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm: bool = True,
        layer_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))
        return x

    def forward(self, x: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, freq_embed)
        else:
            return self._forward(x, freq_embed)

class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            x = module(x, *args, **kwargs)
        return self[-1](x)

@register
class MvHeat_DET(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list = [2,2,9,2],
        dims: Optional[list] = None,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
        post_norm: bool = True,
        layer_scale: Optional[float] = None,
        use_checkpoint: bool = False,
        mlp_ratio: float = 4.0,
        img_size: int = 224,
        act_layer: str = 'GELU',
        infer_mode: bool = False,
        **kwargs,
    ):
        super().__init__()
        dims = dims or [96,192,384,768]
        self.num_classes = num_classes
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.depths = depths

        # stem
        self.patch_embed = StemLayer(in_chans=in_chans, out_chans=self.embed_dim,
                                     act_layer=act_layer, norm_layer='LN')
        # resolution per stage (ceil division to match conv stem downsampling)
        res0 = math.ceil(img_size / patch_size)
        self.res = [
            res0,
            math.ceil(res0 / 2),
            math.ceil(res0 / 4),
            math.ceil(res0 / 8),
        ]

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.infer_mode = infer_mode

        # frequency embeddings
        self.freq_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(r, r, d), requires_grad=True)
            for r,d in zip(self.res, dims)
        ])
        for p in self.freq_embed:
            trunc_normal_(p, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i,(r,d,depth) in enumerate(zip(self.res, dims, depths)):
            dp_rates = dpr[sum(depths[:i]):sum(depths[:i+1])]
            down = (  
                nn.Identity() if i==len(depths)-1 else
                nn.Sequential(nn.Conv2d(d, dims[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                              LayerNorm2d(dims[i+1]))
            )
            layer = self.make_layer(
                res=r, dim=d, depth=depth, drop_path=dp_rates,
                use_checkpoint=use_checkpoint, norm_layer=LayerNorm2d,
                post_norm=post_norm, layer_scale=layer_scale,
                downsample=down, mlp_ratio=mlp_ratio, infer_mode=infer_mode
            )
            self.layers.append(layer)

        # classifier head (unused when num_classes=0)
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def make_layer(res, dim, depth, drop_path, use_checkpoint, norm_layer,
                   post_norm, layer_scale, downsample, mlp_ratio, infer_mode):
        blocks = []
        for dp in drop_path:
            blocks.append(HeatBlock(
                res=res, infer_mode=infer_mode, hidden_dim=dim,
                drop_path=dp, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio, post_norm=post_norm,
                layer_scale=layer_scale
            ))
        return AdditionalInputSequential(*blocks, downsample)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)

    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_heat2d(self.freq_embed[i])
        del self.freq_embed

    def forward(self, x: torch.Tensor):
        # only features, no classifier
        x = self.patch_embed(x)
        for i,layer in enumerate(self.layers):
            x = layer(x, None if self.infer_mode else self.freq_embed[i])
        return x

# --- TrackerNetHeat integrating both branches ---
@register
class TrackerNetHeat(Template):
    def __init__(
        self,
        representation: str = "time_surfaces_1",
        max_unrolls: int = 16,
        n_vis: int = 8,
        feature_dim: int = 1024,
        patch_size: int = 31,
        init_unrolls: int = 1,
        target_channels: int = 6,
        heat_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(representation, max_unrolls, init_unrolls, n_vis, patch_size, **kwargs)
        self.feature_dim = feature_dim
        # Lazy init placeholders
        self.target_channels = target_channels
        self.reference_channels = None
        self.reference_encoder = None
        self.target_encoder = None
        self.reference_redir = None
        self.target_redir = None
        self.heat_backbone_t = None
        self.heat_backbone_r = None
        self.heat_redir = None
        self.f_ref = None
        self.h_ref = None
        # Store for lazy init
        self._heat_kwargs = heat_kwargs or {}
        self._patch_size = patch_size
        # Ensure MvHeat_DET resolution aligns: patch_size must be divisible by 4
        # (Removed static patch_size divisibility assertion)
        # Fusion & predictor
        self.softmax           = nn.Softmax(dim=2)
        joint_channels         = 2 + 4*128
        joint_dim              = joint_channels * (patch_size**2)
        self.joint_mlp         = nn.Sequential(
            nn.Linear(joint_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.predictor         = nn.Linear(512, 2, bias=False)
        # Loss and name
        self.loss              = L1Truncated(patch_size)
        self.name              = f"heat+spatial_corr_{self.representation}"

    def _lazy_init(self, x: torch.Tensor):
        total_c = x.shape[1]
        c_t = self.target_channels
        assert total_c > c_t, f"Total channels {total_c} must exceed target_channels {c_t}"
        c_r = total_c - c_t
        self.reference_channels = c_r

        # Spatial backbones
        self.target_encoder = FPNEncoder(c_t, self.feature_dim, recurrent=True)
        self.reference_encoder = FPNEncoder(c_r, self.feature_dim, recurrent=True)
        self.target_redir = nn.Conv2d(self.feature_dim, 128, kernel_size=3, padding=1)
        self.reference_redir = nn.Conv2d(self.feature_dim, 128, kernel_size=3, padding=1)

        # Heat backbones use fixed patch embed of 4x downsampling
        H = x.shape[2]
        self.heat_backbone_t = MvHeat_DET(
            patch_size=4,
            in_chans=c_t,
            num_classes=0,
            infer_mode=False,
            img_size=H,
            **self._heat_kwargs
        )
        self.heat_backbone_r = MvHeat_DET(
            patch_size=4,
            in_chans=c_r,
            num_classes=0,
            infer_mode=False,
            img_size=H,
            **self._heat_kwargs
        )
        self.heat_redir = nn.Conv2d(self.heat_backbone_t.num_features, 128, kernel_size=1)

        # Move newly created modules to the correct device
        device = x.device
        self.target_encoder = self.target_encoder.to(device)
        self.reference_encoder = self.reference_encoder.to(device)
        self.target_redir = self.target_redir.to(device)
        self.reference_redir = self.reference_redir.to(device)
        self.heat_backbone_t = self.heat_backbone_t.to(device)
        self.heat_backbone_r = self.heat_backbone_r.to(device)
        self.heat_redir = self.heat_redir.to(device)

        # Reset caches
        self.f_ref = None
        self.h_ref = None

    def reset(self, _batch=None):
        if self.reference_encoder is not None:
            self.reference_encoder.reset()
        if self.target_encoder is not None:
            self.target_encoder.reset()
        self.f_ref = None
        self.h_ref = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        if self.reference_encoder is None:
            self._lazy_init(x)
        # x: [B, C_total, H, W]
        c_t = self.target_channels
        c_r = self.reference_channels
        x_t = x[:, :c_t, :, :]
        x_r = x[:, c_t:c_t + c_r, :, :]
        # Spatial features
        f_t, _ = self.target_encoder(x_t)
        f_t    = self.target_redir(f_t)
        # Get spatial resolution from spatial branch
        H_s, W_s = f_t.shape[2], f_t.shape[3]
        if self.f_ref is None:
            f_r, _    = self.reference_encoder(x_r)
            self.f_ref = self.reference_redir(f_r)
        # Heat features
        h_t = self.heat_backbone_t(x_t)
        h_t = self.heat_redir(h_t)
        # Upsample heat features to match spatial resolution
        h_t = F.interpolate(h_t, size=(H_s, W_s), mode='bilinear', align_corners=False)
        if self.h_ref is None:
            h_r = self.heat_backbone_r(x_r)
            self.h_ref = self.heat_redir(h_r)
            # Upsample reference heat features to match spatial resolution
            self.h_ref = F.interpolate(self.h_ref, size=(H_s, W_s), mode='bilinear', align_corners=False)
        # Correlations
        corr_s = (f_t * self.f_ref).sum(dim=1, keepdim=True)
        corr_h = (h_t * self.h_ref).sum(dim=1, keepdim=True)
        # Spatial correlation normalization
        B, _, H_s, W_s = corr_s.shape
        corr_s = self.softmax(corr_s.view(B, 1, H_s * W_s)).view(B, 1, H_s, W_s)
        # Heat correlation normalization and expand to spatial resolution
        B, _, H_h, W_h = corr_h.shape
        corr_h = self.softmax(corr_h.view(B, 1, H_h * W_h)).view(B, 1, H_h, W_h)
        corr_h = corr_h.expand(-1, -1, H_s, W_s)
        # Joint fusion & prediction
        joint    = torch.cat([corr_s, f_t, self.f_ref, corr_h, h_t, self.h_ref], dim=1)
        joint_vec = joint.view(B, -1)
        feat     = self.joint_mlp(joint_vec)
        out      = self.predictor(feat)
        return out