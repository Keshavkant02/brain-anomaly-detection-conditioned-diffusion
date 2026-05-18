import logging
import warnings
from typing import List, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
# Using the PL 1.6.5 naming convention
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)
    for level in ("debug", "info", "warning", "error", "exception", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger

log = get_logger(__name__)

def extras(config: DictConfig) -> None:
    """Applies optional utilities, usually for debugging."""
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

@rank_zero_only
def print_config(config: DictConfig, resolve: bool = True) -> None:
    """Prints content of DictConfig using Rich library and can resolve interpolations."""
    import rich.syntax
    import rich.tree
    from rich.console import Console

    style = "bold magenta"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        else:
            branch_content = str(config_section)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    Console().print(tree)

def log_hyperparameters(
    config: DictConfig,
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    callbacks: List[Callback],
    logger: List[LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers."""
    hparams = {}

    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # --- Robust Logger ID Handling for PL 1.6.5 ---
    if trainer.logger:
        try:
            # For CSVLogger, the 'version' is the folder name (e.g., version_0)
            hparams["run_id"] = getattr(trainer.logger, "version", "local_run")
            if hasattr(trainer.logger, "id"):
                hparams["run_id"] = trainer.logger.id
        except Exception:
            hparams["run_id"] = "local_run"

    # Save hyperparameters
    trainer.logger.log_hyperparams(hparams)

def finish(
    config: DictConfig,
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    callbacks: List[Callback],
    logger: List[LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""
    for lg in logger:
        if isinstance(lg, LightningLoggerBase) and hasattr(lg, "experiment") and hasattr(lg.experiment, "finish"):
            lg.experiment.finish()

def summarize(metrics_dict, prefix):
    """Summarizes metrics for logging."""
    new_dict = {}
    for key, value in metrics_dict.items():
        if isinstance(value, list):
            new_dict[prefix + "/" + key] = sum(value) / len(value)
        else:
            new_dict[prefix + "/" + key] = value
    return new_dict

def get_checkpoint(cfg, checkpoint_path):
    """Helper to resolve checkpoint paths for resuming."""
    return "local_id", {"fold-1": checkpoint_path}
