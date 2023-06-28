import os.path as path
import pickle
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from . import utility as util


class _BaseModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self) -> optim.SGD:
        return optim.SGD(self.parameters(), lr=0.1)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        estim = self(batch[0])
        loss = self.criterion(estim, batch[1])
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor, _: int) -> None:
        estim = self(batch[0])
        loss = self.criterion(estim, batch[1])
        self.log("validation_loss", loss)

    def on_test_start(self) -> None:
        self.test_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def test_step(self, batch: torch.Tensor, _: int) -> None:
        self.test_outputs.append((self(batch[0]), batch[1]))

    def on_test_end(self) -> None:
        estim = np.empty((0, 10), dtype=np.float32)
        truth = np.empty(0, dtype=np.int32)
        for o in self.test_outputs:
            estim = np.vstack((estim, o[0].cpu().numpy()))
            truth = np.hstack((truth, o[1].squeeze().cpu().numpy()))

        with open(path.join(self.logger.log_dir, "test_outputs.pkl"), mode="wb") as f:
            pickle.dump((estim, truth), f)

    def on_predict_start(self) -> None:
        self.predict_outputs = []

    def predict_step(self, patch: torch.Tensor, _: int) -> torch.Tensor:
        estim = self(patch)
        self.predict_outputs.append(estim)

        return estim

    def on_predict_end(self) -> None:
        estim = torch.stack(self.predict_outputs).cpu().numpy()

        with open(path.join(self.logger.log_dir, "predict_outputs.pkl"), mode="wb") as f:
            pickle.dump((self.trainer.predict_dataloaders.dataset.cam_name, self.trainer.predict_dataloaders.dataset.vid_idx, self.trainer.predict_dataloaders.dataset.img.numpy(), estim, self.trainer.predict_dataloaders.dataset.label), f)
        util.write_predict_result(self.trainer.predict_dataloaders.dataset.cam_name, self.trainer.predict_dataloaders.dataset.vid_idx, estim, self.logger.log_dir)

class CNN(_BaseModule):
    def __init__(self, param: dict[str, int]) -> None:
        super().__init__()

        self.save_hyperparameters(param)

        self.conv_1 = nn.Conv2d(3, param["conv_1_ch"], param["conv_ks"])
        self.conv_2 = nn.Conv2d(param["conv_1_ch"], param["conv_2_ch"], param["conv_ks"])
        self.fc = nn.Linear((24 - 2 * param["conv_ks"]) * (19 - 2 * param["conv_ks"]) * param["conv_2_ch"], 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, channel)
        hidden = F.dropout(F.relu(self.conv_1(input)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), training=self.training)
        output = self.fc(hidden.flatten(start_dim=1))

        return output

class FullNet(_BaseModule):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(1122, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, channel)
        output = self.layer1(input.flatten(start_dim=1))

        return output
