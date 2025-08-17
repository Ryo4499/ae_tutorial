from argparse import Namespace
import torch
from torch.nn import functional as F
from torch import nn, Tensor
from pytorch_lightning import LightningModule

from .utils.entity import SupportedArchitecture


class AE(LightningModule):
    def __init__(
        self,
        args: Namespace,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.architecture = args.architecture
        self.dataset = args.dataset
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor):
        h = self.encoder(x)
        output = self.decoder(h)
        return output, h

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch: tuple[Tensor], batch_idx: int):
        x, _ = batch
        if self.architecture == SupportedArchitecture.FAA:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        x_hat = self.decoder(h)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor], batch_idx: int):
        x, _ = batch
        if self.architecture == SupportedArchitecture.FAA:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        x_hat = self.decoder(h)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def test_step(self, batch: tuple[Tensor], batch_idx: int):
        x, _ = batch
        if self.architecture == SupportedArchitecture.FAA:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        x_hat = self.decoder(h)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)
