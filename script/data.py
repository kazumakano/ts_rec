import os.path as path
from datetime import timedelta
from glob import glob, iglob
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import functional as TF
from tqdm import tqdm
from . import utility as util


class TsFigDataset(data.Dataset):
    def __init__(self, files: list[str], aug_num: int = 64, brightness: float = 0.2, contrast: float = 0.2, hue: float = 0.2, max_shift_len: int = 4, norm: bool = False) -> None:
        self.aug_num = aug_num

        self.img = torch.empty((self.aug_num * len(files), 3, 22, 17), dtype=torch.float32)
        self.label = torch.empty(len(files), dtype=torch.int64)
        for i, f in enumerate(tqdm(files, desc="loading timestamp figure images")):
            self.img[self.aug_num * i:self.aug_num * i + self.aug_num] = util.aug_img(TF.to_tensor(Image.open(f)), self.aug_num, brightness, contrast, hue, max_shift_len)
            self.label[i] = int(f[-5])
        if norm:
            self.img = TF.normalize(self.img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.img[idx], self.label[idx // self.aug_num]

    def __len__(self) -> int:
        return len(self.img)

    @property
    def breakdown(self) -> np.ndarray:
        breakdown = np.empty(10, dtype=np.int32)
        for i in range(10):
            breakdown[i] = len(self.label[self.label == i])

        return breakdown

class VidDataset(data.Dataset):
    def __init__(self, files: list[str], norm: bool = False, sec_per_file: float = 1791, show_progress: bool = True) -> None:
        self.cam_name = np.empty(len(files), dtype="<U3")
        self.vid_idx = np.empty(len(files), dtype=np.int32)
        self.img = torch.empty((len(files), 6, 3, 22, 17), dtype=torch.float32)
        self.label = np.empty(len(files), dtype=timedelta)
        for i, f in enumerate(tqdm(files, desc="loading videos", disable=not show_progress)):
            self.cam_name[i] = path.basename(path.dirname(f))[6:]
            self.vid_idx[i] = int(f[-6:-4])
            for j, tmp_img in enumerate(util.extract_ts_fig(util.read_head_n_frms(f, 1).squeeze())):
                self.img[i, j] = TF.to_tensor(tmp_img)
            if norm:
                self.img[i] = TF.normalize(self.img[i], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            self.label[i] = util.calc_ts_from_name(f, sec_per_file)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.img[idx // 6, idx % 6]

    def __len__(self) -> int:
        return 6 * len(self.label)

class DataModule(pl.LightningDataModule):
    def __init__(self, param: dict[str | util.Param], ts_fig_dir: Optional[list[str]] = None, vid_dir: Optional[str] = None, ex_file: Optional[str] = None, seed: int = 0) -> None:
        super().__init__()

        self.dataset: dict[str, TsFigDataset | VidDataset] = {}
        self.save_hyperparameters(param)

        if ts_fig_dir is not None:
            files = []
            for d in ts_fig_dir:
                files += glob(path.join(d, "*_[0-9].tif"))
            self.train_files, self.val_files, self.test_files = util.random_split(files, (0.8, 0.1, 0.1), seed)

        if vid_dir is not None:
            exclude = None if ex_file is None else util.load_param(ex_file)
            self.predict_files = []
            for d in sorted(iglob(path.join(vid_dir, "camera*"))):
                if exclude is None or exclude["camera"] is None or path.basename(d)[6:] not in exclude["camera"]:
                    for f in sorted(iglob(path.join(d, "video_??-??-??_??.mkv"))):
                        if exclude is None or exclude["index"] is None or int(f[-6:-4]) not in exclude["index"]:
                            self.predict_files.append(f)

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                if "train" not in self.dataset.keys():
                    self.dataset["train"] = TsFigDataset(self.train_files)
                    self.dataset["validate"] = TsFigDataset(self.val_files, 1)
            case "test":
                self.dataset["test"] = TsFigDataset(self.test_files, 1)
            case "predict":
                self.dataset["predict"] = VidDataset(self.predict_files)

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"], shuffle=self.hparams["shuffle"], num_workers=self.hparams["num_workers"])

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["validate"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["predict"], batch_size=6, num_workers=self.hparams["num_workers"])
