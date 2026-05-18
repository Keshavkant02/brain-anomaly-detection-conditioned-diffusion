import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim as optim
from typing import Any
import torchio as tio
import numpy as np

from src.models.modules.spark.Spark_3D import SparK_3D as SparK_Model
from src.models.losses import L1_AE
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary


class Spark_3D(LightningModule):
    def __init__(self, cfg, prefix=None):
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix if prefix is not None else ''

        self.model = SparK_Model(cfg)
        self.L1 = L1_AE(cfg)

        self.cond_proj = nn.Linear(2048, cfg.get('cond_dim', 128))
        self.save_hyperparameters(ignore=['cfg'])

    def forward(self, x):
        active_ex, reco, spatial_loss, latent = self.model(x)

        if self.cfg.get('loss_on_mask', False):
            loss = spatial_loss
        else:
            loss = self.L1({'x_hat': reco}, x)['recon_error'] \
                   + self.cfg.get('delta_mask', 0) * spatial_loss

        context_vector = self.cond_proj(latent[0].mean([2, 3, 4]))
        return loss, reco, context_vector

    def training_step(self, batch, batch_idx: int):
        input_vol = batch['vol'][tio.DATA].permute(0, 1, 4, 2, 3)
        loss, reco, _ = self(input_vol)
        self.log(f'{self.prefix}train/Loss_comb', loss,
                 prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=input_vol.shape[0], sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        input_vol = batch['vol'][tio.DATA].permute(0, 1, 4, 2, 3)
        loss, reco, _ = self(input_vol)
        self.log(f'{self.prefix}val/Loss_comb', loss,
                 prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=input_vol.shape[0], sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.new_size = [160, 190, 160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self, 'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        self.dataset = batch['Dataset']
        input_vol = batch['vol'][tio.DATA].permute(0, 1, 4, 2, 3)
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        self.stage = batch['stage']
        label = batch['label']

        loss, reco_vol, _ = self(input_vol)
        final_volume = reco_vol.squeeze().permute(1, 2, 0)
        AnomalyScoreReco_vol = loss.item()

        self.eval_dict['AnomalyScoreRegPerVol'].append(0)
        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(0)
            self.eval_dict['AnomalyScoreCombiPerVol'].append(0)
            self.eval_dict['AnomalyScoreCombPriorPerVol'].append(0)
            self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(0)

        final_volume = final_volume.unsqueeze(0).unsqueeze(0)
        _test_step(self, final_volume, data_orig, data_seg, data_mask,
                   batch_idx, ID, label)

    def on_test_end(self):
        _test_end(self)

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.cfg.get('lr', 1e-4),
            weight_decay=self.cfg.get('weight_decay', 0.05),
            betas=[0.9, 0.95]
        )

    def on_save_checkpoint(self, checkpoint):
        enc_state = {}
        for k, v in self.state_dict().items():
            if k.startswith('model.dense_encoder.'):
                new_k = k.replace('model.dense_encoder.', '')
                enc_state[new_k] = v
            elif k.startswith('cond_proj.'):
                enc_state[k] = v
        checkpoint['encoder_state_dict'] = enc_state

    def update_prefix(self, prefix):
        self.prefix = prefix
