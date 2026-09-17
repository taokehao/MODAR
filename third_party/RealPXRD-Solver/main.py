import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks import Callback
from typing import List
from pathlib import Path
import warnings
import time
import torch.multiprocessing as MProcess
from app.data.utils import build_callbacks, Wandb_logger
warnings.filterwarnings('ignore')
MProcess.set_sharing_strategy('file_system')

PROJECT_ROOT = Path('Your code path')

@hydra.main(config_path=str(PROJECT_ROOT/'conf'), config_name="default")
def main(cfg: DictConfig):
    start = time.time()
    hydra_dir = Path(HydraConfig.get().run.dir)
    if cfg.logging.reproducibility.determine:
        seed_everything(cfg.logging.reproducibility.seed)
    
    dataModule: L.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, optim=cfg.optim, _recursive_=False)
    callbacks: List[Callback] = build_callbacks(cfg=cfg)
    
    wandb_logger = Wandb_logger(cfg=cfg)
    wandb_logger.watch(model, log=cfg.logging.wandb_watch.log, log_freq=cfg.logging.wandb_watch.log_freq)
    trainer = L.Trainer(
        strategy='ddp_find_unused_parameters_true',
        default_root_dir=hydra_dir,
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=cfg.logging.reproducibility.determine,
        **cfg.logging.pl_trainer
    )
    
    hydra.utils.log.info("Start training!")
    trainer.fit(model=model, datamodule=dataModule)
    hydra.utils.log.info("Start testing!")
    trainer.test(datamodule=dataModule, ckpt_path='last')

    end = time.time()
    use_time = (end-start)/3600
    hydra.utils.log.info("Using time %s h"%use_time)


if __name__ == '__main__':
    main()


