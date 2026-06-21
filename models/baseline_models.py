"""
Faithful re-implementations of SRCNN, BasicVSR, and EDVR that load mmagic
pretrained checkpoints directly without mmcv/mmagic installed.

Pretrained weights auto-download from OpenMMLab on first use.

Checkpoints used
----------------
SRCNN    : srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth
BasicVSR : basicvsr_vimeo90k_bi_20210409-d2d8f760.pth  (+ spynet)
EDVR-M   : edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth  (5-frame, REDS)

Key-structure compatibility
---------------------------
* mmagic wraps every net in a `generator` attribute → strip 'generator.' prefix.
* mmcv.ConvModule stores conv as `.conv`           → our _ConvModule matches.
* mmcv.ModulatedDeformConv2d weight/offset layout  → torchvision.ops.deform_conv2d
  uses identical shapes (offset_groups == deform_groups).
* mmagic.PixelShufflePack stores upsample conv as `.upsample_conv`.
* mmagic.ResidualBlockNoBN: conv1, conv2, relu.
"""

import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ─── Checkpoint registry & download ──────────────────────────────────────────

CKPT_DIR = Path('checkpoints/baselines')

_URLS = {
    'srcnn': (
        'srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
        'https://download.openmmlab.com/mmediting/restorers/srcnn/'
        'srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
    ),
    'basicvsr': (
        'basicvsr_vimeo90k_bi_20210409-d2d8f760.pth',
        'https://download.openmmlab.com/mmediting/restorers/basicvsr/'
        'basicvsr_vimeo90k_bi_20210409-d2d8f760.pth',
    ),
    'spynet': (
        'spynet_20210409-c6c1bd09.pth',
        'https://download.openmmlab.com/mmediting/restorers/basicvsr/'
        'spynet_20210409-c6c1bd09.pth',
    ),
    'edvr': (
        'edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth',
        'https://download.openmmlab.com/mmediting/restorers/edvr/'
        'edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth',
    ),
}


def _download(name: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    filename, url = _URLS[name]
    path = CKPT_DIR / filename
    if not path.exists():
        print(f'[baseline] Downloading {name} …  {url}')
        def _hook(count, block, total):
            pct = min(count * block / total * 100, 100)
            sys.stdout.write(f'\r  {pct:5.1f}%')
            sys.stdout.flush()
        urllib.request.urlretrieve(url, path, _hook)
        print(f'\n  → {path}')
    return path


def _load_sd(path: Path, device, strip_prefix: str = 'generator.') -> dict:
    """Load a checkpoint and strip the given key prefix."""
    raw = torch.load(str(path), map_location=device, weights_only=False)
    sd  = raw.get('state_dict', raw)
    return {
        (k[len(strip_prefix):] if k.startswith(strip_prefix) else k): v
        for k, v in sd.items()
    }


# ─── Shared building blocks (key-compatible with mmagic) ─────────────────────

class _ConvModule(nn.Module):
    """
    mmcv.cnn.ConvModule replacement.
    Stores Conv2d as `self.conv` → checkpoint keys like `*.conv.weight` match.
    Activation has no parameters → not in checkpoint, handled in forward().
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 stride: int = 1, padding: int = 0, act: str = 'lrelu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self._act = act  # 'lrelu' | 'relu' | 'none'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self._act == 'lrelu':
            return F.leaky_relu(x, 0.1, inplace=True)
        if self._act == 'relu':
            return F.relu(x, inplace=True)
        return x


class _ResBlockNoBN(nn.Module):
    """mmagic.models.archs.ResidualBlockNoBN — keys: conv1, conv2, relu."""
    def __init__(self, mid_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class _PixelShufflePack(nn.Module):
    """mmagic.models.archs.PixelShufflePack — key: upsample_conv."""
    def __init__(self, in_ch: int, out_ch: int, scale: int, kernel: int = 3):
        super().__init__()
        self.upsample_conv = nn.Conv2d(in_ch, out_ch * scale * scale, kernel,
                                       padding=(kernel - 1) // 2)
        self.pixel_shuffle  = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.upsample_conv(x))


def _make_layer(block, n: int, **kw) -> nn.Sequential:
    return nn.Sequential(*[block(**kw) for _ in range(n)])


# ─── SRCNN ────────────────────────────────────────────────────────────────────

class SRCNN(nn.Module):
    """
    mmagic SRCNNNet — keys: conv1, conv2, conv3, img_upsampler, relu.
    Single-image SR via bicubic upscale + 3-layer refinement.

    Input:  center_lr  (B, 3, H_lr, W_lr)
    Output: SR image   (B, 3, H_hr, W_hr)
    """

    def __init__(self, scale: int = 4):
        super().__init__()
        self.upscale_factor = scale
        self.img_upsampler  = nn.Upsample(scale_factor=scale, mode='bicubic',
                                           align_corners=False)
        self.conv1 = nn.Conv2d(3,  64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32,  3, kernel_size=5, padding=2)
        self.relu  = nn.ReLU()

    @classmethod
    def from_pretrained(cls, device, user_ckpt: str = '') -> 'SRCNN':
        model = cls().to(device)
        path  = Path(user_ckpt) if user_ckpt else _download('srcnn')
        model.load_state_dict(_load_sd(path, device), strict=True)
        return model.eval()

    def forward(self, center_lr: torch.Tensor, *_) -> torch.Tensor:
        x = self.img_upsampler(center_lr)
        return self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))).clamp(0, 1)


# ─── BasicVSR ─────────────────────────────────────────────────────────────────
# Full architecture matching mmagic BasicVSRNet.
# SPyNet weights are loaded separately (spynet_20210409-c6c1bd09.pth).

def _flow_warp(x: torch.Tensor, flow: torch.Tensor,
               mode: str = 'bilinear', padding: str = 'zeros') -> torch.Tensor:
    """Warp x with optical flow (n, h, w, 2) using grid_sample."""
    n, _, h, w = x.size()
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=x.dtype, device=x.device),
        torch.arange(w, dtype=x.dtype, device=x.device),
        indexing='ij',
    )
    grid  = torch.stack((gx, gy), 2).unsqueeze(0)   # (1, h, w, 2)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((vgrid_x, vgrid_y), 3),
                         mode=mode, padding_mode=padding, align_corners=True)


class _SPyNetBasic(nn.Module):
    """SPyNetBasicModule — 5 ConvModules, keys: basic_module.{0-4}.conv.*"""
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvModule(8,  32, 7, padding=3, act='relu'),
            _ConvModule(32, 64, 7, padding=3, act='relu'),
            _ConvModule(64, 32, 7, padding=3, act='relu'),
            _ConvModule(32, 16, 7, padding=3, act='relu'),
            _ConvModule(16,  2, 7, padding=3, act='none'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basic_module(x)


class _SPyNet(nn.Module):
    """SPyNet — keys: basic_module.{0-5}, mean, std (buffers)."""
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([_SPyNetBasic() for _ in range(6)])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.size()
        ref_  = [(ref  - self.mean) / self.std]
        supp_ = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_.append(F.avg_pool2d(ref_[-1],  2, 2, count_include_pad=False))
            supp_.append(F.avg_pool2d(supp_[-1], 2, 2, count_include_pad=False))
        ref_  = ref_[::-1]
        supp_ = supp_[::-1]

        flow = ref_[0].new_zeros(n, 2, h // 32, w // 32)
        for lvl in range(6):
            flow_up = flow if lvl == 0 else (
                F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            )
            flow = flow_up + self.basic_module[lvl](
                torch.cat([ref_[lvl],
                           _flow_warp(supp_[lvl], flow_up.permute(0, 2, 3, 1),
                                      padding='border'),
                           flow_up], 1)
            )
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w   = ref.shape[2:]
        w_up   = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up   = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref_u  = F.interpolate(ref,  (h_up, w_up), mode='bilinear', align_corners=False)
        supp_u = F.interpolate(supp, (h_up, w_up), mode='bilinear', align_corners=False)
        flow   = F.interpolate(self._compute_flow(ref_u, supp_u), (h, w),
                               mode='bilinear', align_corners=False)
        flow[:, 0] *= float(w) / w_up
        flow[:, 1] *= float(h) / h_up
        return flow


class _ResBlocksWithInputConv(nn.Module):
    """mmagic ResidualBlocksWithInputConv — keys: main.0.weight, main.2.{i}.conv{1,2}.*"""
    def __init__(self, in_ch: int, out_ch: int = 64, n: int = 30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            _make_layer(_ResBlockNoBN, n, mid_channels=out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class BasicVSR(nn.Module):
    """
    mmagic BasicVSRNet — full architecture with SPyNet optical-flow alignment.

    Checkpoints: basicvsr_vimeo90k_bi_20210409-d2d8f760.pth
                 spynet_20210409-c6c1bd09.pth   (loaded separately)

    Input:  lr_frames  (B, 7, 3, H_lr, W_lr)
    Output: SR centre  (B, 3, H_hr, W_hr)  — im4
    """

    def __init__(self, mid: int = 64, n_blocks: int = 30):
        super().__init__()
        self.mid_channels   = mid
        self.spynet          = _SPyNet()
        self.backward_resblocks = _ResBlocksWithInputConv(mid + 3, mid, n_blocks)
        self.forward_resblocks  = _ResBlocksWithInputConv(mid + 3, mid, n_blocks)
        self.fusion       = nn.Conv2d(mid * 2, mid, 1, 1, 0)
        self.upsample1    = _PixelShufflePack(mid, mid, 2, 3)
        self.upsample2    = _PixelShufflePack(mid, 64,  2, 3)
        self.conv_hr      = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last    = nn.Conv2d(64,  3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu        = nn.LeakyReLU(0.1, inplace=True)
        self._warned      = False

    @classmethod
    def from_pretrained(cls, device, user_ckpt: str = '', **_) -> 'BasicVSR':
        model   = cls().to(device)
        # BasicVSR checkpoint already contains all SPyNet weights (including mean/std)
        path    = Path(user_ckpt) if user_ckpt else _download('basicvsr')
        missing, unexpected = model.load_state_dict(_load_sd(path, device), strict=False)
        if missing:
            print(f'[BasicVSR] missing keys ({len(missing)}): {missing[:3]}')
        return model.eval()

    def forward(self, lr_frames: torch.Tensor, *_) -> torch.Tensor:
        n, t, c, h, w = lr_frames.size()
        if (h < 64 or w < 64) and not self._warned:
            print(f'[BasicVSR] warning: input {h}×{w} is smaller than 64×64')
            self._warned = True

        # Compute pairwise optical flows
        lrs_1 = lr_frames[:, :-1].reshape(-1, c, h, w)
        lrs_2 = lr_frames[:, 1: ].reshape(-1, c, h, w)
        flows_bwd = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_fwd = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        # Backward propagation
        bwd_out = []
        fp = lr_frames.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                fp = _flow_warp(fp, flows_bwd[:, i].permute(0, 2, 3, 1))
            fp = self.backward_resblocks(torch.cat([lr_frames[:, i], fp], 1))
            bwd_out.append(fp)
        bwd_out = bwd_out[::-1]

        # Forward propagation + upsample for each frame
        fp = torch.zeros_like(fp)
        upsampled = []
        for i in range(t):
            lr_i = lr_frames[:, i]
            if i > 0:
                fp = _flow_warp(fp, flows_fwd[:, i - 1].permute(0, 2, 3, 1))
            fp = self.forward_resblocks(torch.cat([lr_i, fp], 1))

            out = self.lrelu(self.fusion(torch.cat([bwd_out[i], fp], 1)))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out) + self.img_upsample(lr_i)
            upsampled.append(out)

        return upsampled[t // 2].clamp(0, 1)   # centre frame (im4)


# ─── EDVR ─────────────────────────────────────────────────────────────────────
# EDVR-M with TSA, trained on REDS x4 (5 input frames, center_frame_idx=2).
# We feed it frames [1..5] of our 7-frame sequence so im4 stays the centre.

class _ModulatedDCNPack(nn.Module):
    """
    mmcv ModulatedDCNPack replacement using torchvision.ops.deform_conv2d.

    Weight shapes are identical:
      weight      (out_ch, in_ch, kH, kW)
      conv_offset  predicts (B, 2*dg*kH*kW + dg*kH*kW, H, W) → split to offset + mask
    torchvision infers offset_groups from offset.shape[1] / (2*kH*kW).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 padding: int = 0, deform_groups: int = 8):
        super().__init__()
        self.padding      = padding
        self.deform_groups = deform_groups
        k = kernel

        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias   = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight)

        # offset + mask from extra_feat (matches mmagic key 'conv_offset.*')
        self.conv_offset = nn.Conv2d(
            in_ch,
            deform_groups * 3 * k * k,   # 2*dg*k^2 offsets + dg*k^2 masks
            kernel_size=k, padding=padding,
        )
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: torch.Tensor, extra_feat: torch.Tensor) -> torch.Tensor:
        out    = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask   = torch.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, self.bias,
                             padding=self.padding, mask=mask)


class _PCDAlignment(nn.Module):
    """mmagic PCDAlignment (3-level pyramid + cascading DCN)."""

    def __init__(self, mid: int = 64, deform_groups: int = 8):
        super().__init__()
        # nn.ModuleDict preserves string keys → checkpoint keys: 'l1','l2','l3'
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack     = nn.ModuleDict()
        self.feat_conv    = nn.ModuleDict()

        for i in range(3, 0, -1):
            lv = f'l{i}'
            self.offset_conv1[lv] = _ConvModule(mid * 2, mid, 3, padding=1)
            if i == 3:
                self.offset_conv2[lv] = _ConvModule(mid, mid, 3, padding=1)
            else:
                self.offset_conv2[lv] = _ConvModule(mid * 2, mid, 3, padding=1)
                self.offset_conv3[lv] = _ConvModule(mid, mid, 3, padding=1)
            self.dcn_pack[lv] = _ModulatedDCNPack(mid, mid, 3, padding=1,
                                                    deform_groups=deform_groups)
            if i < 3:
                act = 'lrelu' if i == 2 else 'none'
                self.feat_conv[lv] = _ConvModule(mid * 2, mid, 3, padding=1, act=act)

        self.cas_offset_conv1 = _ConvModule(mid * 2, mid, 3, padding=1)
        self.cas_offset_conv2 = _ConvModule(mid,     mid, 3, padding=1)
        self.cas_dcnpack      = _ModulatedDCNPack(mid, mid, 3, padding=1,
                                                   deform_groups=deform_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu    = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, nbr: List[torch.Tensor], ref: List[torch.Tensor]) -> torch.Tensor:
        up_offset = up_feat = None
        for i in range(3, 0, -1):
            lv = f'l{i}'
            offset = self.offset_conv1[lv](torch.cat([nbr[i - 1], ref[i - 1]], 1))
            if i == 3:
                offset = self.offset_conv2[lv](offset)
            else:
                offset = self.offset_conv2[lv](torch.cat([offset, up_offset], 1))
                offset = self.offset_conv3[lv](offset)

            feat = self.dcn_pack[lv](nbr[i - 1], offset)
            feat = self.lrelu(feat) if i == 3 else self.feat_conv[lv](
                torch.cat([feat, up_feat], 1))

            if i > 1:
                up_offset = self.upsample(offset) * 2
                up_feat   = self.upsample(feat)

        offset = self.cas_offset_conv2(self.cas_offset_conv1(torch.cat([feat, ref[0]], 1)))
        return self.lrelu(self.cas_dcnpack(feat, offset))


class _TSAFusion(nn.Module):
    """mmagic TSAFusion — temporal + spatial attention fusion."""

    def __init__(self, mid: int = 64, num_frames: int = 5, center: int = 2):
        super().__init__()
        self.center_frame_idx = center
        self.temporal_attn1   = nn.Conv2d(mid, mid, 3, padding=1)
        self.temporal_attn2   = nn.Conv2d(mid, mid, 3, padding=1)
        self.feat_fusion      = _ConvModule(num_frames * mid, mid, 1)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1    = _ConvModule(num_frames * mid, mid, 1)
        self.spatial_attn2    = _ConvModule(mid * 2, mid, 1)
        self.spatial_attn3    = _ConvModule(mid, mid, 3, padding=1)
        self.spatial_attn4    = _ConvModule(mid, mid, 1)
        self.spatial_attn5    = nn.Conv2d(mid, mid, 3, padding=1)
        self.spatial_attn_l1  = _ConvModule(mid, mid, 1)
        self.spatial_attn_l2  = _ConvModule(mid * 2, mid, 3, padding=1)
        self.spatial_attn_l3  = _ConvModule(mid, mid, 3, padding=1)
        self.spatial_attn_add1 = _ConvModule(mid, mid, 1)
        self.spatial_attn_add2 = nn.Conv2d(mid, mid, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu    = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, aligned: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = aligned.size()
        emb_ref = self.temporal_attn1(aligned[:, self.center_frame_idx])
        emb     = self.temporal_attn2(aligned.view(-1, c, h, w)).view(n, t, c, h, w)
        corr    = torch.sigmoid(torch.cat(
            [(emb[:, i] * emb_ref).sum(1, keepdim=True) for i in range(t)], 1
        ))   # (n, t, h, w)
        corr    = corr.unsqueeze(2).expand(n, t, c, h, w).reshape(n, -1, h, w)
        af      = aligned.view(n, -1, h, w) * corr

        feat    = self.feat_fusion(af)
        attn    = self.spatial_attn1(af)
        attn    = self.spatial_attn2(torch.cat([self.max_pool(attn), self.avg_pool(attn)], 1))
        al      = self.spatial_attn_l1(attn)
        al      = self.spatial_attn_l2(torch.cat([self.max_pool(al), self.avg_pool(al)], 1))
        al      = self.upsample(self.spatial_attn_l3(al))
        attn    = self.upsample(self.spatial_attn4(self.spatial_attn3(attn) + al))
        attn    = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        return feat * torch.sigmoid(attn) * 2 + attn_add


class EDVR(nn.Module):
    """
    mmagic EDVRNet-M with TSA, REDS ×4.

    Checkpoint : edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth
    num_frames : 5  (center_frame_idx=2)
    We select frames [1:6] from our 7-frame input so im4 stays the centre.

    Input:  lr_frames  (B, 7, 3, H_lr, W_lr)
    Output: SR centre  (B, 3, H_hr, W_hr)
    """

    def __init__(self, mid: int = 64, num_frames: int = 5,
                 deform_groups: int = 8, n_ext: int = 5, n_rec: int = 10):
        super().__init__()
        self.center = num_frames // 2   # 2

        self.conv_first        = nn.Conv2d(3, mid, 3, 1, 1)
        self.feature_extraction = _make_layer(_ResBlockNoBN, n_ext, mid_channels=mid)

        self.feat_l2_conv1 = _ConvModule(mid, mid, 3, stride=2, padding=1)
        self.feat_l2_conv2 = _ConvModule(mid, mid, 3, padding=1)
        self.feat_l3_conv1 = _ConvModule(mid, mid, 3, stride=2, padding=1)
        self.feat_l3_conv2 = _ConvModule(mid, mid, 3, padding=1)

        self.pcd_alignment = _PCDAlignment(mid, deform_groups)
        self.fusion        = _TSAFusion(mid, num_frames, self.center)

        self.reconstruction = _make_layer(_ResBlockNoBN, n_rec, mid_channels=mid)
        self.upsample1  = _PixelShufflePack(mid, mid, 2, 3)
        self.upsample2  = _PixelShufflePack(mid, 64,  2, 3)
        self.conv_hr    = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last  = nn.Conv2d(64,  3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    @classmethod
    def from_pretrained(cls, device, user_ckpt: str = '') -> 'EDVR':
        model = cls().to(device)
        path  = Path(user_ckpt) if user_ckpt else _download('edvr')
        sd    = _load_sd(path, device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f'[EDVR] missing keys ({len(missing)}): {missing[:3]} …')
        return model.eval()

    def forward(self, lr_frames: torch.Tensor, *_) -> torch.Tensor:
        # Use centre 5 frames so im4 (index 3 in 7-frame) → index 2 in 5-frame
        x = lr_frames[:, 1:6]          # (B, 5, 3, H, W)
        n, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, f"H,W must be divisible by 4, got {h},{w}"

        x_center = x[:, self.center].contiguous()

        # Multi-scale feature extraction
        l1 = self.lrelu(self.conv_first(x.reshape(-1, c, h, w)))
        l1 = self.feature_extraction(l1)
        l2 = self.feat_l2_conv2(self.feat_l2_conv1(l1))
        l3 = self.feat_l3_conv2(self.feat_l3_conv1(l2))

        l1 = l1.reshape(n, t, -1, h,      w     )
        l2 = l2.reshape(n, t, -1, h // 2, w // 2)
        l3 = l3.reshape(n, t, -1, h // 4, w // 4)

        ref = [l1[:, self.center].clone(),
               l2[:, self.center].clone(),
               l3[:, self.center].clone()]

        aligned = torch.stack([
            self.pcd_alignment(
                [l1[:, i].clone(), l2[:, i].clone(), l3[:, i].clone()], ref
            ) for i in range(t)
        ], dim=1)   # (n, t, c, h, w)

        feat = self.fusion(aligned)
        feat = self.reconstruction(feat)
        out  = self.lrelu(self.upsample1(feat))
        out  = self.lrelu(self.upsample2(out))
        out  = self.lrelu(self.conv_hr(out))
        out  = self.conv_last(out) + self.img_upsample(x_center)
        return out.clamp(0, 1)
