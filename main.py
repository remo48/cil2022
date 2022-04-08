import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import RoadSegDataModule, DummyDataset
from src.models.UNet import SMPModel
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import DataLoader


def get_dummy_dataloader(num_samples, batch_size=8, num_workers=4):
    ds = DummyDataset(num_samples)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

def run_experiment():
    seed_everything(1234)
    encoder = "mobilenet_v2"

    datamodule = RoadSegDataModule(
        data_dir="data")
    model = SMPModel("FPN", encoder, 3, 1)
    # model = SMPModel.load_from_checkpoint("logs/default/version_1/checkpoints/epoch=9-step=169.ckpt")
    logger = TensorBoardLogger(save_dir="logs")

    trainer = Trainer(max_epochs=30, logger=logger, log_every_n_steps=5)
    trainer.fit(model, datamodule=datamodule)
    # trainer.fit(model, train_dataloaders=get_dummy_dataloader(num_samples=180), val_dataloaders=get_dummy_dataloader(num_samples=20))


if __name__ == "__main__":
    run_experiment()
