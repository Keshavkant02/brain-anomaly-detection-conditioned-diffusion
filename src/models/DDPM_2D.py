from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.DDPM_encoder import get_encoder
import torch
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any
import torchio as tio
from src.utils.generate_noise import gen_noise
import wandb
from omegaconf import open_dict
from collections import OrderedDict
from src.models.LDM.modules.diffusionmodules.util import timestep_embedding
import torch.nn as nn


class DDPM_2D(LightningModule):
    def __init__(self, cfg, prefix=None):
        super().__init__()

        self.cfg = cfg

        # ---------------------------------------------------------
        # ENCODER / CONDITIONING SETUP
        # ---------------------------------------------------------
        if cfg.get('precomputed_cond', False):
            # 3D path: offline latents, project 2048 -> 128 inside DDPM
            self.encoder = None
            self.cond_proj = nn.Linear(2048, cfg.get('cond_dim', 128))
            out_features = cfg.get('cond_dim', 128)
            print(f"[DDPM_2D] Using precomputed 3D latents from {cfg.get('latent_dir', 'cache/latents_gold700')}")
        else:
            # 2D path: online encoder (Finn's original)
            if cfg.get('condition', True):
                with open_dict(self.cfg):
                    self.cfg['cond_dim'] = cfg.get('unet_dim', 64) * 4
                self.encoder, out_features = get_encoder(cfg)
            else:
                out_features = None

        # ---------------------------------------------------------
        # UNet
        # ---------------------------------------------------------
        model = OpenAI_UNet(
            image_size=(int(cfg.imageDim[0] / cfg.rescaleFactor), int(cfg.imageDim[1] / cfg.rescaleFactor)),
            in_channels=1,
            model_channels=cfg.get('unet_dim', 64),
            out_channels=1,
            num_res_blocks=cfg.get('num_res_blocks', 3),
            attention_resolutions=tuple(cfg.get('att_res', [3, 6, 12])),
            dropout=cfg.get('dropout_unet', 0),
            channel_mult=cfg.get('dim_mults', [1, 2, 4, 8]),
            conv_resample=True,
            dims=2,
            num_classes=out_features,
            use_checkpoint=False,
            use_fp16=True,
            num_heads=1,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
            use_spatial_transformer=cfg.get('spatial_transformer', False),
            transformer_depth=1,
        )
        model.convert_to_fp16()

        timesteps = cfg.get('timesteps', 1000)
        sampling_timesteps = cfg.get('sampling_timesteps', timesteps)
        self.test_timesteps = cfg.get('test_timesteps', 150)

        self.diffusion = GaussianDiffusion(
            model,
            image_size=(int(cfg.imageDim[0] / cfg.rescaleFactor), int(cfg.imageDim[1] / cfg.rescaleFactor)),
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=cfg.get('objective', 'pred_x0'),
            channels=1,
            loss_type=cfg.get('loss', 'l1'),
            p2_loss_weight_gamma=cfg.get('p2_gamma', 0),
            cfg=cfg
        )

        # ---------------------------------------------------------
        # PRETRAINED ENCODER LOADER
        # ---------------------------------------------------------
        if cfg.get('precomputed_cond', False):
            # cond_proj trains from scratch with DDPM; no pretrained loading needed
            pass
        elif cfg.get('pretrained_encoder', False):
            print('Loading pretrained encoder from: ', cfg.encoder_path)
            assert cfg.get('encoder_path', None) is not None

            ckpt = torch.load(cfg.get('encoder_path', None), map_location='cpu')

            # 3D encoder path: clean export from Spark_3D
            if 'encoder_state_dict' in ckpt:
                state_dict = ckpt['encoder_state_dict']
                proj_shape = state_dict.get('cond_proj.weight', torch.empty(0)).shape
                if proj_shape:
                    print(f'[Encoder load] Found cond_proj: {proj_shape[0]}d output')
                self.encoder.load_state_dict(state_dict, strict=True)
                print('[Encoder load] Loaded from encoder_state_dict (strict)')

            # Legacy 2D path: Finn's original remapping
            else:
                state_dict_pretrained = ckpt['state_dict']
                new_statedict = OrderedDict()
                for key in zip(state_dict_pretrained):
                    k = key[0]
                    if 'slice_encoder' in k:
                        new_key = 'slice_encoder' + k.split('encoder')[-1]
                        new_statedict[new_key] = state_dict_pretrained[k]
                    elif 'sparse_encoder' in k:
                        if 'fc.weight' not in k and 'fc.bias' not in k:
                            new_key = 'encoder' + k.split('sp_cnn')[-1]
                            new_statedict[new_key] = state_dict_pretrained[k]
                    else:
                        new_statedict[k] = state_dict_pretrained[k]
                self.encoder.load_state_dict(new_statedict, strict=False)
                print('[Encoder load] Loaded from legacy state_dict (non-strict)')

        self.prefix = prefix
        self.save_hyperparameters()

    def forward(self, x):
        if self.cfg.get('precomputed_cond', False):
            # x is already the latent vector [B, 2048]
            return self.cond_proj(x)
        elif self.cfg.get('condition', True):
            x = self.encoder(x)
        else:
            x = None
        return x

    def _get_features(self, batch, input_img):
        """Unified feature extraction for train/val/test."""
        if self.cfg.get('precomputed_cond', False):
            # batch['cond'] is [B, 1, 2048, 1] from TorchIO ScalarImage
            cond = batch['cond'][tio.DATA].squeeze()  # [B, 2048]
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            features = self.cond_proj(cond.to(self.device))  # [B, 128]
        else:
            features = self(input_img)
        return features

    def training_step(self, batch, batch_idx: int):
        input = batch['vol'][tio.DATA].squeeze(-1)
        features = self._get_features(batch, input)

        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else:
            noise = None

        loss, reco = self.diffusion(input, cond=features, noise=noise)

        self.log(f'{self.prefix}train/Loss', loss, prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=input.shape[0], sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        input = batch['vol'][tio.DATA].squeeze(-1)
        features = self._get_features(batch, input)

        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else:
            noise = None

        loss, reco = self.diffusion(input, cond=features, noise=noise)

        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=input.shape[0], sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160, 190, 160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self, 'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        age = batch['age']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []

        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4):
            num_slices = self.cfg.get('num_eval_slices', input.size(4))
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[..., start_slice:start_slice + num_slices]
            data_orig = data_orig[..., start_slice:start_slice + num_slices]
            data_seg = data_seg[..., start_slice:start_slice + num_slices]
            data_mask = data_mask[..., start_slice:start_slice + num_slices]
            ind_offset = start_slice
        else:
            ind_offset = 0

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device=self.device)

        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3, 0, 1, 2)  # [B,C,H,W,D] -> [D,C,H,W]

        # Feature extraction
        features = self._get_features(batch, input)
        features_single = features

        if self.cfg.condition:
            latentSpace.append(features_single.mean(0).squeeze().detach().cpu())
        else:
            latentSpace.append(torch.tensor([0], dtype=float).repeat(input.shape[0]))

        if self.cfg.get('noise_ensemble', False):
            timesteps = self.cfg.get('step_ensemble', [250, 500, 750])
            reco_ensemble = torch.zeros_like(input)
            for t in timesteps:
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, input.shape).to(self.device)
                else:
                    noise = None
                loss_diff, reco = self.diffusion(input, cond=features, t=t - 1, noise=noise)
                reco_ensemble += reco
            reco = reco_ensemble / len(timesteps)
        else:
            if self.cfg.get('noisetype') is not None:
                noise = gen_noise(self.cfg, input.shape).to(self.device)
            else:
                noise = None
            loss_diff, reco = self.diffusion(input, cond=features, t=self.test_timesteps - 1, noise=noise)

        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(loss_diff.cpu())
        AnomalyScoreReco.append(loss_diff.cpu())

        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1, 2, 0)  # to HxWxD

        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace), 0))
        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb)
        AnomalyScoreReg_vol = np.mean(AnomalyScoreReg)
        AnomalyScoreReco_vol = np.mean(AnomalyScoreReco)

        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)

        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)
            self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol * AnomalyScoreReg_vol)
            self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol + self.cfg.beta * 0)
            self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol * 0)

        final_volume = final_volume.unsqueeze(0).unsqueeze(0)
        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label)

    def on_test_end(self):
        _test_end(self)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)

    def update_prefix(self, prefix):
        self.prefix = prefix
