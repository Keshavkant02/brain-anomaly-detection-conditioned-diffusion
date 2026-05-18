import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)

_BN = None

class UNetBlock3x(nn.Module):
    def __init__(self, cin, cout, cmid, last_act=True):
        super().__init__()
        c_mid = cin if cmid == 0 else (cin + cout) // 2
            
        self.b = nn.Sequential(
            nn.Conv3d(cin, c_mid, 3, 1, 1, bias=False), 
            _BN(c_mid), 
            nn.ReLU6(inplace=True),
            nn.Conv3d(c_mid, cout, 3, 1, 1, bias=False), 
            _BN(cout), 
            (nn.ReLU6(inplace=True) if last_act else nn.Identity()),
        )
        
    def forward(self, x):
        return self.b(x)

class DecoderConv3D(nn.Module):
    def __init__(self, cin, cout, double, heavy, cmid):
        super().__init__()
        # 3D Transposed Conv for Volumetric Upsampling
        self.up = nn.ConvTranspose3d(
            cin, cin, 
            kernel_size=4 if double else 2, 
            stride=2, 
            padding=1 if double else 0, 
            bias=True
        )
        ls = [UNetBlock3x(cin, (cin if i != heavy[1]-1 else cout), cmid=cmid, last_act=i != heavy[1]-1) for i in range(heavy[1])]
        self.conv = nn.Sequential(*ls)
    
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class LightDecoder3D(nn.Module):
    def __init__(self, decoder_fea_dim, upsample_ratio, double=False, heavy=None, cmid=0, sbn=False):
        global _BN
        _BN = nn.SyncBatchNorm if sbn else nn.BatchNorm3d
        super().__init__()
        self.fea_dim = decoder_fea_dim
        self.heavy = heavy if heavy is not None else [0, 1]
        
        # Calculate hierarchical channels
        n = round(math.log2(upsample_ratio))
        channels = [self.fea_dim // 2**i for i in range(n+1)]
        
        self.dec = nn.ModuleList([
            DecoderConv3D(cin, cout, double, self.heavy, cmid) 
            for (cin, cout) in zip(channels[:-1], channels[1:])
        ])
        
        # Final projection to a single 3D channel (The MRI Intensity)
        self.proj = nn.Conv3d(channels[-1], 1, kernel_size=1, stride=1, bias=True)
        self.initialize()
    
    def forward(self, to_dec):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
