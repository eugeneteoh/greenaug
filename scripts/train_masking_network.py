import hydra
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from greenaug.masking_network import MaskingDataset, MaskingNetwork


@hydra.main(config_path="../conf", config_name="mask", version_base=None)
def main(cfg):
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    seed_everything(cfg.seed, workers=True)

    dataset = MaskingDataset(cfg.preprocessed_root)

    train_ratio = 0.75
    train_len = round(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = hydra.utils.instantiate(cfg.model, _target_=MaskingNetwork)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        save_top_k=-1,
        every_n_epochs=5,
        monitor="val/loss",
        verbose=True,
        save_last=True,
    )

    wandb_logger = WandbLogger(
        **dict(cfg.wandb),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        logger=wandb_logger,
        log_every_n_steps=cfg.log_every_n_steps,
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
