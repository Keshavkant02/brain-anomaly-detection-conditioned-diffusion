import random
import torch
import torch.nn as nn
from typing import List

from monai.networks.nets import ResNet
from src.models.modules.spark.decoder import LightDecoder3D


class ResNet50_3D_Backbone(nn.Module):
    """
    Flattened MONAI ResNet50 that returns hierarchical feature maps.
    Uses string 'bottleneck' for version-agnostic MONAI compatibility.
    """
    def __init__(self):
        super().__init__()
        resnet = ResNet(
            block='bottleneck',  # string — MONAI handles ResNetBottleneck internally
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=0,
        )
        self.conv1 = resnet.conv1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


class SparK_3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mask_ratio = cfg.get('mask_ratio', 0.65)
        self.mask_ratio = (mask_ratio, mask_ratio)
        self.uniform = cfg.get('uniform', False)
        self.pix_norm = int(cfg.get('pix_norm', 1))
        self.dense_loss = cfg.get('dense_loss', False)
        self.loss_l2 = cfg.get('loss_l2', True)
        self.pyramid = cfg.get('pyramid', 4)

        # 1. DENSE 3D BACKBONE
        self.dense_encoder = ResNet50_3D_Backbone()
        self.downsample_raito = 16

        # 2. 3D DECODER
        self.dense_decoder = LightDecoder3D(
            cfg.get('dec_dim', 512),
            self.downsample_raito,
            double=cfg.get('double', True)
        )

        # 3. CHANNEL ADAPTERS & MASK TOKENS
        self.en_de_norms = nn.ModuleList()
        self.en_de_lins = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        # layer4=2048, layer3=1024, layer2=512, layer1=256
        encoder_channels = [256, 512, 1024, 2048]
        encoder_channels.reverse()          # deepest first
        d_fea = self.dense_decoder.fea_dim  # 512

        for i in range(self.pyramid):
            fea = encoder_channels[i]
            self.en_de_norms.append(nn.BatchNorm3d(fea))

            kernel_size = 1 if i <= 0 else 3
            l = nn.Conv3d(fea, d_fea, kernel_size=kernel_size, stride=1,
                          padding=kernel_size // 2, bias=True)
            if i == 0 and fea == d_fea:
                l = nn.Identity()
            self.en_de_lins.append(l)

            p = nn.Parameter(torch.zeros(1, fea, 1, 1, 1))
            nn.init.trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)

            d_fea //= 2   # 512 → 256 → 128 → 64

    def mask(self, shape, device, generator=None):
        """Generates the 3D volumetric boolean mask."""
        B, C, D, H, W = shape
        p = self.downsample_raito
        d, h, w = D // p, H // p, W // p

        L = d * h * w
        len_keep = round(L * (1 - self.mask_ratio[0]))

        idx = torch.rand(B, L, generator=generator).argsort(dim=1)
        idx = idx[:, :len_keep].to(device)

        mask = torch.zeros(B, L, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True)
        return mask.view(B, 1, d, h, w)

    def patchify(self, bcdhw):
        """[B, C, D, H, W] → [B, d*h*w, p³·C]"""
        p = self.downsample_raito
        B, C, D, H, W = bcdhw.shape
        d, h, w = D // p, H // p, W // p

        bcdhw = bcdhw.reshape(shape=(B, C, d, p, h, p, w, p))
        bcdhw = bcdhw.permute(0, 2, 4, 6, 3, 5, 7, 1)
        bln = bcdhw.reshape(shape=(B, d * h * w, (p ** 3) * C))
        return bln

    def forward(self, raw_inp: torch.Tensor, active=None):
        orig_D, orig_H, orig_W = raw_inp.shape[2], raw_inp.shape[3], raw_inp.shape[4]
        # Dynamic pad to multiple of downsample_raito
        B, C, D, H, W = raw_inp.shape
        p = self.downsample_raito
        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_d or pad_h or pad_w:
            raw_inp = torch.nn.functional.pad(raw_inp, (0, pad_w, 0, pad_h, 0, pad_d))
        B, C, D, H, W = raw_inp.shape

        # 1. Volumetric masking
        if active is None:
            active: torch.BoolTensor = self.mask(raw_inp.shape, raw_inp.device)

        active_ex = active.repeat_interleave(self.downsample_raito, 2) \
                           .repeat_interleave(self.downsample_raito, 3) \
                           .repeat_interleave(self.downsample_raito, 4)
        masked_bcdhw = raw_inp * active_ex

        # 2. Extract hierarchical dense features
        fea_bcffs = self.dense_encoder(masked_bcdhw)   # [layer1, layer2, layer3, layer4]
        fea_bcffs.reverse()                             # smallest → largest

        # 3. Channel adapters & mask tokens
        cur_active = active
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):
            if bcff is not None:
                bcff = self.en_de_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff).type_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)
                bcff = self.en_de_lins[i](bcff)
            to_dec.append(bcff)

            cur_active = cur_active.repeat_interleave(2, dim=2) \
                                   .repeat_interleave(2, dim=3) \
                                   .repeat_interleave(2, dim=4)

        # 4. Decode
        rec_bcdhw = self.dense_decoder(to_dec)

        # 5. Patchified spatial loss (masked patches only)
        inp_patches = self.patchify(raw_inp)
        rec_patches = self.patchify(rec_bcdhw)

        if self.pix_norm == 1:
            mean = inp_patches.mean(dim=-1, keepdim=True)
            var = (inp_patches.var(dim=-1, keepdim=True) + 1e-6) ** .5
            inp_patches = (inp_patches - mean) / var

        loss_spa = (rec_patches - inp_patches) ** 2 if self.loss_l2 else (rec_patches - inp_patches).abs()
        loss_spa = loss_spa.mean(dim=2, keepdim=False)
        non_active = active.logical_not().int().view(B, -1)
        spatial_loss = loss_spa.mul_(non_active).sum() / (non_active.sum() + 1e-8)

        rec_bcdhw = rec_bcdhw[:, :, :orig_D, :orig_H, :orig_W]
        return active_ex, rec_bcdhw, spatial_loss, fea_bcffs
