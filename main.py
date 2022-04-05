import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import RoadSegDataModule
from src.models.UNet import PretrainedUNet, SMPModel
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def run_experiment():
    seed_everything(1234)
    encoder = "resnet34"
    encoder_weights = "imagenet"

    datamodule = RoadSegDataModule(
        data_dir="data")
    model = SMPModel("Unet", encoder, 3, 1)
    logger = TensorBoardLogger(save_dir="logs")

    trainer = Trainer(max_epochs=5, logger=logger, log_every_n_steps=5)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    run_experiment()
