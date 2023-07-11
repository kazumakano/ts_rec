import os.path as path
import pickle
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from . import utility as util


class _BaseModule(pl.LightningModule):
    def __init__(self, loss_weight: torch.Tensor | None) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(weight=loss_weight)

    def configure_optimizers(self) -> optim.SGD:
        return optim.SGD(self.parameters(), lr=self.hparams["learning_rate"])

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
        self.test_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def test_step(self, batch: torch.Tensor, _: int) -> None:
        self.test_outputs.append((batch[0], self(batch[0]), batch[1]))

    def on_test_end(self) -> None:
        img = np.empty((0, 3, 22, 17), dtype=np.float32)
        estim = np.empty((0, 10), dtype=np.float32)
        truth = np.empty(0, dtype=np.int32)
        for o in self.test_outputs:
            img = np.vstack((img, o[0].cpu().numpy()))
            estim = np.vstack((estim, o[1].cpu().numpy()))
            truth = np.hstack((truth, o[2].squeeze().cpu().numpy()))

        with open(path.join(self.logger.log_dir, "test_outputs.pkl"), mode="wb") as f:
            pickle.dump((img, estim, truth), f)

    def on_predict_start(self) -> None:
        self.predict_outputs = []

    def predict_step(self, patch: torch.Tensor, _: int) -> torch.Tensor:
        estim = self(patch)
        self.predict_outputs.append(estim)

        return estim

    def on_predict_end(self) -> None:
        estim = torch.stack(self.predict_outputs).cpu().numpy()

        outputs_file = path.join(self.logger.log_dir, "predict_outputs.pkl")
        if path.exists(outputs_file):
            with open(outputs_file, mode="rb") as f:
                pre_cam_name, pre_vid_idx, pre_img, pre_estim, pre_label = pickle.load(f)
            with open(outputs_file, mode="wb") as f:
                pickle.dump((
                    np.hstack((pre_cam_name, self.trainer.predict_dataloaders.dataset.cam_name)),
                    np.hstack((pre_vid_idx, self.trainer.predict_dataloaders.dataset.vid_idx)),
                    np.vstack((pre_img, self.trainer.predict_dataloaders.dataset.img.numpy())),
                    np.vstack((pre_estim, estim)),
                    np.hstack((pre_label, self.trainer.predict_dataloaders.dataset.label))
                ), f)
        else:
            with open(outputs_file, mode="wb") as f:
                pickle.dump((
                    self.trainer.predict_dataloaders.dataset.cam_name,
                    self.trainer.predict_dataloaders.dataset.vid_idx,
                    self.trainer.predict_dataloaders.dataset.img.numpy(),
                    estim,
                    self.trainer.predict_dataloaders.dataset.label
                ), f)
        util.write_predict_result(self.trainer.predict_dataloaders.dataset.cam_name, self.trainer.predict_dataloaders.dataset.vid_idx, util.get_most_likely_ts(estim), self.trainer.predict_dataloaders.dataset.label, self.logger.log_dir)

class CNN2(_BaseModule):
    def __init__(self, param: dict[str, int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight)

        self.save_hyperparameters(param)

        self.conv_1 = nn.Conv2d(3, param["conv_ch_1"], param["conv_ks_1"])
        self.conv_2 = nn.Conv2d(param["conv_ch_1"], param["conv_ch_2"], param["conv_ks_2"])
        self.fc = nn.Linear((24 - param["conv_ks_1"] - param["conv_ks_2"]) * (19 - param["conv_ks_1"] - param["conv_ks_2"]) * param["conv_ch_2"], 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class)
        hidden = F.dropout(F.relu(self.conv_1(input)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), training=self.training)
        output = self.fc(hidden.flatten(start_dim=1))

        return output

class CNN3(_BaseModule):
    def __init__(self, param: dict[str, int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight)

        self.save_hyperparameters(param)

        self.conv_1 = nn.Conv2d(3, param["conv_ch_1"], param["conv_ks_1"])
        self.conv_2 = nn.Conv2d(param["conv_ch_1"], param["conv_ch_2"], param["conv_ks_2"])
        self.conv_3 = nn.Conv2d(param["conv_ch_2"], param["conv_ch_3"], param["conv_ks_3"])
        self.fc = nn.Linear((25 - param["conv_ks_1"] - param["conv_ks_2"] - param["conv_ks_3"]) * (20 - param["conv_ks_1"] - param["conv_ks_2"] - param["conv_ks_3"]) * param["conv_ch_3"], 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class)
        hidden = F.dropout(F.relu(self.conv_1(input)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_3(hidden)), training=self.training)
        output = self.fc(hidden.flatten(start_dim=1))

        return output

    @staticmethod
    def is_valid_ks(param: dict[str | util.Param]) -> bool:
        return param["conv_ks_1"] + param["conv_ks_2"] + param["conv_ks_3"] < 20

class FullNet(_BaseModule):
    def __init__(self, loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight)

        self.layer1 = nn.Sequential(
            nn.Linear(1122, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class)
        output = self.layer1(input.flatten(start_dim=1))

        return output
