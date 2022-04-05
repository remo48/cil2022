import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import RoadSegDataModule
from src.models.UNet import CustomUNet


MODEL_DICT = {
    "baseline_unet": CustomUNet
}

def run_experiment():
    seed_everything(1234)

    datamodule = RoadSegDataModule(data_dir="data")
    model = CustomUNet()
    logger = TensorBoardLogger(save_dir="logs")

    trainer = Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    run_experiment()