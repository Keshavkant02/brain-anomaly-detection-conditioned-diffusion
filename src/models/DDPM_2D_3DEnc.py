"""
DDPM_2D_3DEnc.py

Stage 2 model:
- pretrained 3D Spark/ResNet50 encoder
- 2D conditional DDPM UNet
- encoder produces 128-d conditioning vector from full 3D MRI volume
- DDPM reconstructs 2D slices conditioned on 3D anatomical context

This file implements the final 3D-context conditioned DDPM workflow
remains untouched.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio

from typing import Any
from pytorch_lightning.core.lightning import LightningModule

from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.spark.Spark_3D import ResNet50_3D_Backbone
from src.utils.generate_noise import gen_noise


class DDPM_2D_3DEnc(LightningModule):
    def __init__(self, cfg, prefix=None):
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix or ""

        # ---------------------------------------------------------
        # 3D encoder
        # ---------------------------------------------------------
        self.encoder_3d = ResNet50_3D_Backbone()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cond_proj = nn.Linear(2048, cfg.get("cond_dim", 128))

        encoder_path = cfg.get("encoder_path", None)
        if encoder_path is None or str(encoder_path).strip() in ["", "???", "None"]:
            raise ValueError(
                "model.cfg.encoder_path is missing. Pass it on the command line, e.g. "
                "model.cfg.encoder_path=/path/to/epoch-...ckpt"
            )

        ckpt = torch.load(encoder_path, map_location="cpu")

        if "encoder_state_dict" in ckpt:
            enc_state = ckpt["encoder_state_dict"]
        else:
            # Fallback: strip possible Lightning prefixes.
            enc_state = {}
            for k, v in ckpt.get("state_dict", ckpt).items():
                if k.startswith("model.dense_encoder."):
                    enc_state[k.replace("model.dense_encoder.", "")] = v
                elif k.startswith("dense_encoder."):
                    enc_state[k.replace("dense_encoder.", "")] = v

        # Clean checkpoint state:
        # Stage-1 checkpoints may contain cond_proj.* keys inside encoder_state_dict.
        # Those are not part of ResNet50_3D_Backbone, so we filter them before strict loading.
        raw_enc_state = dict(enc_state)
        valid_encoder_state = self.encoder_3d.state_dict()

        enc_state = {
            k: v for k, v in raw_enc_state.items()
            if k in valid_encoder_state and tuple(v.shape) == tuple(valid_encoder_state[k].shape)
        }

        dropped = sorted(set(raw_enc_state.keys()) - set(enc_state.keys()))

        missing, unexpected = self.encoder_3d.load_state_dict(enc_state, strict=True)

        # If the Stage-1 checkpoint has a compatible cond_proj, load it too.
        if "cond_proj_state_dict" in ckpt:
            cp_state = ckpt["cond_proj_state_dict"]
            if (
                "weight" in cp_state
                and "bias" in cp_state
                and tuple(cp_state["weight"].shape) == tuple(self.cond_proj.weight.shape)
                and tuple(cp_state["bias"].shape) == tuple(self.cond_proj.bias.shape)
            ):
                self.cond_proj.load_state_dict(cp_state, strict=True)
                print("[DDPM_2D_3DEnc] Loaded cond_proj_state_dict.")

        elif "cond_proj.weight" in raw_enc_state and "cond_proj.bias" in raw_enc_state:
            if (
                tuple(raw_enc_state["cond_proj.weight"].shape) == tuple(self.cond_proj.weight.shape)
                and tuple(raw_enc_state["cond_proj.bias"].shape) == tuple(self.cond_proj.bias.shape)
            ):
                self.cond_proj.load_state_dict(
                    {
                        "weight": raw_enc_state["cond_proj.weight"],
                        "bias": raw_enc_state["cond_proj.bias"],
                    },
                    strict=True,
                )
                print("[DDPM_2D_3DEnc] Loaded cond_proj from encoder_state_dict extras.")

        print(f"[DDPM_2D_3DEnc] Loaded encoder from: {encoder_path}")
        if dropped:
            print(f"[DDPM_2D_3DEnc] Dropped non-encoder keys: {dropped}")
        print(f"[DDPM_2D_3DEnc] Encoder keys loaded: {len(enc_state)}")
        print(f"[DDPM_2D_3DEnc] Missing keys: {len(missing)}")
        print(f"[DDPM_2D_3DEnc] Unexpected keys: {len(unexpected)}")

        if cfg.get("freeze_encoder", False):
            for p in self.encoder_3d.parameters():
                p.requires_grad = False
            print("[DDPM_2D_3DEnc] Encoder frozen.")
        else:
            print("[DDPM_2D_3DEnc] Encoder will be fine-tuned jointly.")

        out_features = cfg.get("cond_dim", 128)

        # ---------------------------------------------------------
        # 2D UNet
        # ---------------------------------------------------------
        image_size = (
            int(cfg.imageDim[0] / cfg.rescaleFactor),
            int(cfg.imageDim[1] / cfg.rescaleFactor),
        )

        unet = OpenAI_UNet(
            image_size=image_size,
            in_channels=1,
            model_channels=cfg.get("unet_dim", 128),
            out_channels=1,
            num_res_blocks=cfg.get("num_res_blocks", 3),
            attention_resolutions=tuple(cfg.get("att_res", [3, 6, 12])),
            dropout=cfg.get("dropout_unet", 0.0),
            channel_mult=cfg.get("dim_mults", [1, 2, 2]),
            conv_resample=cfg.get("conv_resample", True),
            dims=2,
            num_classes=out_features,
            use_checkpoint=False,
            use_fp16=True,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
            use_spatial_transformer=cfg.get("spatial_transformer", False),
            transformer_depth=1,
        )
        unet.convert_to_fp16()

        timesteps = cfg.get("timesteps", 1000)
        sampling_timesteps = cfg.get("sampling_timesteps", timesteps)
        self.test_timesteps = cfg.get("test_timesteps", 150)

        self.diffusion = GaussianDiffusion(
            unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=cfg.get("objective", "pred_x0"),
            channels=1,
            loss_type=cfg.get("loss", "l1"),
            p2_loss_weight_gamma=cfg.get("p2_gamma", 0),
            cfg=cfg,
        )

        self.save_hyperparameters(ignore=["cfg"])

    # ---------------------------------------------------------
    # Tensor helpers
    # ---------------------------------------------------------
    def _extract_tensor(self, obj):
        """Handles TorchIO subject/image dicts and plain tensors."""
        if isinstance(obj, torch.Tensor):
            return obj
        return obj[tio.DATA]

    def _get_volume_3d(self, batch):
        """
        Expected preferred shape from TorchIO:
            [B, C, H, W, D]
        Converted to encoder format:
            [B, C, D, H, W]
        """
        if "vol_3d" in batch:
            x = self._extract_tensor(batch["vol_3d"])
        elif "volume_3d" in batch:
            x = self._extract_tensor(batch["volume_3d"])
        else:
            x = self._extract_tensor(batch["vol"])

        if x.ndim != 5:
            raise ValueError(
                f"3D encoder needs a 5D tensor [B,C,H,W,D] or [B,C,D,H,W], got {tuple(x.shape)}"
            )

        # Your earlier encoder check used permute(0,1,4,2,3), so we assume [B,C,H,W,D].
        if x.shape[-1] > 1:
            return x.permute(0, 1, 4, 2, 3).contiguous()

        raise ValueError(
            f"batch['vol'] appears to contain only a single 2D slice with shape {tuple(x.shape)}. "
            "For real 3D conditioning, the dataloader must provide the full 3D volume."
        )

    def _get_slice_2d(self, batch, random_slice: bool):
        """
        Produces [B,1,H,W] for 2D DDPM.

        If full volume is available [B,C,H,W,D], sample one slice per volume
        during training and use center slice during validation.
        """
        if "slice" in batch:
            x = self._extract_tensor(batch["slice"])
            if x.ndim == 5:
                x = x.squeeze(-1)
            return x

        x = self._extract_tensor(batch["vol"])

        if x.ndim == 4:
            return x

        if x.ndim != 5:
            raise ValueError(f"Cannot create 2D slice from tensor shape {tuple(x.shape)}")

        B, C, H, W, D = x.shape

        if D == 1:
            return x.squeeze(-1)

        if random_slice:
            idx = torch.randint(low=0, high=D, size=(B,), device=x.device)
        else:
            idx = torch.full((B,), D // 2, device=x.device, dtype=torch.long)

        slices = []
        for b in range(B):
            slices.append(x[b, :, :, :, idx[b]])
        return torch.stack(slices, dim=0).contiguous()

    def _encode_3d(self, batch):
        volume_3d = self._get_volume_3d(batch)
        features = self.encoder_3d(volume_3d)
        deepest = features[-1]
        pooled = self.global_pool(deepest).flatten(1)
        cond = self.cond_proj(pooled)
        return cond

    # ---------------------------------------------------------
    # Training / validation
    # ---------------------------------------------------------
    def training_step(self, batch, batch_idx: int):
        slice_2d = self._get_slice_2d(batch, random_slice=True)
        cond = self._encode_3d(batch)

        noise = (
            gen_noise(self.cfg, slice_2d.shape).to(self.device)
            if self.cfg.get("noisetype") is not None
            else None
        )

        loss, reco = self.diffusion(slice_2d, cond=cond, noise=noise)

        self.log(
            f"{self.prefix}train/Loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=slice_2d.shape[0],
            sync_dist=True,
        )
        if self.prefix:
            self.log(
                "train/Loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=slice_2d.shape[0],
                sync_dist=True,
            )

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        slice_2d = self._get_slice_2d(batch, random_slice=False)
        cond = self._encode_3d(batch)

        noise = (
            gen_noise(self.cfg, slice_2d.shape).to(self.device)
            if self.cfg.get("noisetype") is not None
            else None
        )

        loss, reco = self.diffusion(slice_2d, cond=cond, noise=noise)

        self.log(
            f"{self.prefix}val/Loss_comb",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=slice_2d.shape[0],
            sync_dist=True,
        )
        if self.prefix:
            self.log(
                "val/Loss_comb",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=slice_2d.shape[0],
                sync_dist=True,
            )

        return {"loss": loss}

    def forward(self, batch):
        return self._encode_3d(batch)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.cfg.get("lr", 1e-4),
            weight_decay=self.cfg.get("weight_decay", 0.05),
            betas=[0.9, 0.95],
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["encoder_state_dict"] = self.encoder_3d.state_dict()
        checkpoint["cond_proj_state_dict"] = self.cond_proj.state_dict()

    def update_prefix(self, prefix):
        self.prefix = prefix or ""
