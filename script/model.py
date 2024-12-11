import math
import os.path as path
import pickle
import warnings
from typing import Literal, Optional
import easyocr
import numpy as np
import pytorch_lightning as pl
import torch
from easyocr.model.vgg_model import VGG_FeatureExtractor
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from . import utility as util


class _BaseModule(pl.LightningModule):
    def __init__(self, loss_weight: torch.Tensor | None, param: dict[str, util.Param] | None) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(weight=loss_weight)
        if param is not None:
            self.save_hyperparameters(param)

    def configure_optimizers(self) -> optim.SGD:
        return optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def training_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        estim = self(batch[0])
        loss = self.criterion(estim, batch[1])
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: list[torch.Tensor], _: int) -> None:
        estim = self(batch[0])
        loss = self.criterion(estim, batch[1])
        self.log("validation_loss", loss)

    def on_test_start(self) -> None:
        self.test_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def test_step(self, batch: list[torch.Tensor], _: int) -> None:
        self.test_outputs.append((batch[0], self(batch[0]), batch[1]))

    def on_test_end(self) -> None:
        img = np.empty((0, 3, 22, 17), dtype=np.float32)
        estim = np.empty((0, 10), dtype=np.float32)
        truth = np.empty(0, dtype=np.int32)
        for o in self.test_outputs:
            img = np.vstack((img, o[0].cpu().numpy()))
            estim = np.vstack((estim, o[1].cpu().numpy()))
            truth = np.hstack((truth, o[2].squeeze().cpu().numpy()))

        np.savez_compressed(path.join(self.logger.log_dir, "test_outputs.npz"), img=img, estim=estim, truth=truth)

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

        util.write_predict_result(
            self.trainer.predict_dataloaders.dataset.cam_name,
            self.trainer.predict_dataloaders.dataset.vid_idx,
            *util.get_most_likely_ts(estim),
            self.trainer.predict_dataloaders.dataset.label,
            self.trainer.predict_dataloaders.dataset.frm_num,
            self.logger.log_dir
        )

class _BaseModule4ManyFrms(_BaseModule):
    def predict_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        estim = self(batch).reshape(-1, 6, 10)
        self.predict_outputs.append(estim)

        return estim

    def on_predict_end(self) -> None:
        estim = torch.vstack(self.predict_outputs).cpu().numpy()

        estim_ts, conf = util.get_most_likely_ts(estim)
        util.write_predict_result(
            self.trainer.predict_dataloaders.dataset.cam_name,
            self.trainer.predict_dataloaders.dataset.vid_idx,
            estim_ts,
            conf,
            util.check_ts_consis(estim_ts, None if self.trainer.predict_dataloaders.dataset.start_frm_idx == 0 else self.last_estim_ts),
            self.trainer.predict_dataloaders.dataset.start_frm_idx,
            self.logger.log_dir,
        )
        self.last_estim_ts = estim_ts[-1]

        if self.hparams["interp_conf_thresh"] is not None:
            interp_ts, interp_succeeded = util.interp_unconf_ts(estim_ts, conf, self.hparams["interp_conf_thresh"])
            if not interp_succeeded:
                warnings.warn(f"failed to interpolate unconfident timestamps at frame {self.trainer.predict_dataloaders.dataset.start_frm_idx} to {self.trainer.predict_dataloaders.dataset.start_frm_idx + len(interp_ts) - 1} in video {self.trainer.predict_dataloaders.dataset.vid_idx} of camera {self.trainer.predict_dataloaders.dataset.cam_name}")
            util.write_interp_result(
                self.trainer.predict_dataloaders.dataset.cam_name,
                self.trainer.predict_dataloaders.dataset.vid_idx,
                interp_ts,
                util.check_ts_consis(interp_ts, None if self.trainer.predict_dataloaders.dataset.start_frm_idx == 0 else self.last_interp_ts),
                self.trainer.predict_dataloaders.dataset.start_frm_idx,
                self.logger.log_dir
            )
            self.last_interp_ts = interp_ts[-1]

class CNN2(_BaseModule):
    def __init__(self, param: dict[str, int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight, param)

        self.conv_1 = nn.Conv2d(3, param["conv_ch_1"], param["conv_ks_1"])
        self.conv_2 = nn.Conv2d(param["conv_ch_1"], param["conv_ch_2"], param["conv_ks_2"])
        self.fc = nn.Linear((24 - param["conv_ks_1"] - param["conv_ks_2"]) * (19 - param["conv_ks_1"] - param["conv_ks_2"]) * param["conv_ch_2"], 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class)
        hidden = F.dropout(F.relu(self.conv_1(input)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), p=self.hparams["conv_dp"], training=self.training)
        output = self.fc(hidden.flatten(start_dim=1))

        return output

    @staticmethod
    def is_valid_ks(param: dict[str | util.Param]) -> bool:
        return param["conv_ks_1"] + param["conv_ks_2"] < 19

class CNN3(_BaseModule):
    def __init__(self, param: dict[str, int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight, param)

        self.conv_1 = nn.Conv2d(3, param["conv_ch_1"], param["conv_ks_1"])
        self.conv_2 = nn.Conv2d(param["conv_ch_1"], param["conv_ch_2"], param["conv_ks_2"])
        self.conv_3 = nn.Conv2d(param["conv_ch_2"], param["conv_ch_3"], param["conv_ks_3"])
        self.fc = nn.Linear((25 - param["conv_ks_1"] - param["conv_ks_2"] - param["conv_ks_3"]) * (20 - param["conv_ks_1"] - param["conv_ks_2"] - param["conv_ks_3"]) * param["conv_ch_3"], 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class)
        hidden = F.dropout(F.relu(self.conv_1(input)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_3(hidden)), p=self.hparams["conv_dp"], training=self.training)
        output = self.fc(hidden.flatten(start_dim=1))

        return output

    @staticmethod
    def is_valid_ks(param: dict[str | util.Param]) -> bool:
        return param["conv_ks_1"] + param["conv_ks_2"] + param["conv_ks_3"] < 20

class EasyOCR(_BaseModule):
    def __init__(self, param: dict[str, int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight, param)

        self.extractor: VGG_FeatureExtractor = easyocr.Reader(["en"]).recognizer.module.FeatureExtraction
        self.predictor = nn.Sequential(
            nn.Linear(1792, 134),
            nn.ReLU(),
            nn.Linear(134, 10)
        )

        if param["freeze"]:
            for p in self.extractor.parameters():
                p.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class)
        hidden: torch.Tensor = self.extractor(TF.resize(TF.rgb_to_grayscale(input), (self.hparams["scale"] * input.shape[2], self.hparams["scale"] * input.shape[3]), antialias=True))
        hidden = F.adaptive_avg_pool2d(hidden.permute(0, 3, 1, 2), (256, 1))
        output = self.predictor(hidden.flatten(start_dim=1))

        return output

class VGG8(_BaseModule):
    def __init__(self, param: dict[str, int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight, param)

        fc_ch = round(math.sqrt(60 * param["conv_ch_8"]))

        self.conv_1 = nn.Conv2d(3, param["conv_ch_1"], 3)
        self.conv_2 = nn.Conv2d(param["conv_ch_1"], param["conv_ch_2"], 3)
        self.conv_3 = nn.Conv2d(param["conv_ch_2"], param["conv_ch_3"], 3)
        self.conv_4 = nn.Conv2d(param["conv_ch_3"], param["conv_ch_4"], 3)
        self.bn = nn.BatchNorm2d(param["conv_ch_4"])
        self.conv_5 = nn.Conv2d(param["conv_ch_4"], param["conv_ch_5"], 3)
        self.conv_6 = nn.Conv2d(param["conv_ch_5"], param["conv_ch_6"], 3)
        self.conv_7 = nn.Conv2d(param["conv_ch_6"], param["conv_ch_7"], 3)
        self.conv_8 = nn.Conv2d(param["conv_ch_7"], param["conv_ch_8"], 3)
        self.fc_1 = nn.Linear(6 * param["conv_ch_8"], fc_ch)
        self.fc_2 = nn.Linear(fc_ch, 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class)
        hidden = F.dropout(F.relu(self.conv_1(input)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_3(hidden)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.bn(self.conv_4(hidden))), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_5(hidden)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_6(hidden)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_7(hidden)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.dropout(F.relu(self.conv_8(hidden)), p=self.hparams["conv_dp"], training=self.training)
        hidden = F.relu(self.fc_1(hidden.flatten(start_dim=1)))
        output = self.fc_2(hidden)

        return output

class CNN24ManyFrms(_BaseModule4ManyFrms, CNN2):
    ...

class CNN34ManyFrms(_BaseModule4ManyFrms, CNN3):
    ...

class EasyOCR4ManyFrms(_BaseModule4ManyFrms, EasyOCR):
    ...

class VGG84ManyFrms(_BaseModule4ManyFrms, VGG8):
    ...

def get_model_cls(name: Literal["cnn2", "cnn3", "easyocr", "vgg8"], apply_many_frms: bool = False) -> type[CNN2 | CNN24ManyFrms | CNN3 | CNN34ManyFrms | EasyOCR | EasyOCR4ManyFrms | VGG8 | VGG84ManyFrms]:
    match name:
        case "cnn2":
            return CNN24ManyFrms if apply_many_frms else CNN2
        case "cnn3":
            return CNN34ManyFrms if apply_many_frms else CNN3
        case "easyocr":
            return EasyOCR4ManyFrms if apply_many_frms else EasyOCR
        case "vgg8":
            return VGG84ManyFrms if apply_many_frms else VGG8
        case _:
            raise Exception(f"unknown model {name} was specified")
