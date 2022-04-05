from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from pl_bolts.models.vision.unet import UNet


class CustomUNet(LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        num_classes: int = 2,
        num_layers: int = 5,
        features_start: int = 32,
        bilinear: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.net = UNet(
            num_classes=num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument(
            "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
        )

        return parser