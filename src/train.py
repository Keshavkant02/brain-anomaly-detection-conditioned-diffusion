import sys
import os
# This forces the 'src' directory into the search path manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from omegaconf import DictConfig, OmegaConf, open_dict
# For PL 1.6.5, we use the older plugin/logger naming
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import LightningLoggerBase
import hydra
from typing import List, Optional
import os
import warnings
import torch
from src.utils import utils
import pickle

os.environ['NUMEXPR_MAX_THREADS'] = '16'
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

log = utils.get_logger(__name__) 

@hydra.main(config_path='configs', config_name='config') 
def train(cfg: DictConfig) -> Optional[float]: 
    results = {}

    # Base names for logging
    base = cfg.callbacks.model_checkpoint.monitor 
    if 'early_stop' in cfg.callbacks:
        base_es = cfg.callbacks.early_stop.monitor 

    # --- WANDB REMOVED ---
    # We bypass all wandb ID generation and resume logic to avoid Protobuf errors.

    # Set plugins for lightning trainer
    if cfg.trainer.get('accelerator', None) == 'ddp': 
        plugs = DDPPlugin(find_unused_parameters=False)
    else: 
        plugs = None

    if "seed" in cfg: 
        log.info(f"Seed specified to {cfg.seed} by config")
        seed_everything(cfg.seed, workers=True)

    start_fold = cfg.get('start_fold', 0)
    end_fold = cfg.get('num_folds', 1) # Set to 1 for a single Gold_700 run

    for fold in range(start_fold, end_fold): 
        log.info(f"Training Fold {fold+1} of {end_fold}")
        prefix = f'{fold+1}/' 

        # We keep the target specified in your YAML rather than forcing Datamodules_train
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>") 
        datamodule_train: LightningDataModule = hydra.utils.instantiate(cfg.datamodule) 

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model, prefix=prefix) 

        # Setup callbacks
        cfg.callbacks.model_checkpoint.monitor = f'{prefix}' + base 
        cfg.callbacks.model_checkpoint.filename = "epoch-{epoch}_step-{step}_loss-{" + f"{prefix}" + "val/loss:.2f}" 

        if 'early_stop' in cfg.callbacks:
            cfg.callbacks.early_stop.monitor = f'{prefix}' + base_es 

        callbacks: List[Callback] = []
        if "callbacks" in cfg:
            for _, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
        
        if len(callbacks) > 0:
            callbacks[0].FILE_EXTENSION = f'_fold-{fold+1}.ckpt' 

        # Init lightning loggers (CSV Logger)
        logger: List[LightningLoggerBase] = []
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    # Skip WandB logger if it accidentally remains in config
                    if "WandbLogger" in lg_conf._target_:
                        continue
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        # Init lightning trainer
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial", plugins=plugs
        )          

        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=cfg,
            model=model,
            datamodule=datamodule_train,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        if not cfg.get('onlyEval', False): 
            trainer.fit(model, datamodule_train)
        
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    log.info("Finalizing!")
    return None
