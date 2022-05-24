import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.nn import functional as F
from PIL import Image
import os
import albumentations as A
import cv2

from src.dataset import RoadSegDataModule, DummyDataset
from src.UNet import SMPModel
from src.utils import LogPredictionsCallback
from torch.utils.data import DataLoader


def get_dummy_dataloader(num_samples, batch_size=8, num_workers=4):
    ds = DummyDataset(num_samples)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

def save_predictions(preds):
    pred_path = "data/processed/test/predictions"
    os.makedirs(pred_path, exist_ok=True)

    preds = torch.cat(preds, dim=0)
    preds = (preds*255)
    #preds_averaged = F.avg_pool2d(preds, kernel_size=16)
    #preds_imgs = (preds_averaged > 0.25) * 255

    for i, img in enumerate(preds):
        img = img.squeeze().numpy()
        img = A.Resize(400, 400, interpolation=cv2.INTER_NEAREST, always_apply=True)(image=img)["image"]
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        idx = i + 144
        img_path = os.path.join(pred_path, "satimage_" + str(idx) + ".png")
        img.save(img_path)

def run_experiment():
    seed_everything(1234)

    train = True # Set to true, if you want to train the model

    encoder = "resnet34"
    model = SMPModel("FPN", encoder, in_channels=3, out_classes=1)

    datamodule = RoadSegDataModule(
        data_dir="data")

    wandb_logger = WandbLogger(project="cil2022")
    logger = CSVLogger(save_dir="logs")
    loggers = [wandb_logger, logger] if train else []

    log_predictions_callback = LogPredictionsCallback(wandb_logger)
    checkpoint_callback = ModelCheckpoint("logs/checkpoints", monitor="val_f1", mode="max")
    callbacks = [checkpoint_callback]

    trainer = Trainer(
        max_epochs=30,
        default_root_dir="logs",
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=5)

    if train:
        trainer.fit(model, datamodule=datamodule)
    else:
        model = model.load_from_checkpoint("logs/default_cil2022/10_2asjnsgz/checkpoints/epoch=29-step=509.ckpt")

    preds = trainer.predict(model, datamodule=datamodule)
    save_predictions(preds)

if __name__ == "__main__":
    run_experiment()
