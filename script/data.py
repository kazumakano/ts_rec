import os.path as path
from datetime import timedelta
from glob import glob
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import functional as TF
from . import utility as util


class TsFigDataset(data.Dataset):
    def __init__(self, dir: str, aug_num: int = 8, max_shift_len: int = 5) -> None:
        self.aug_num = aug_num

        files = glob(path.join(dir, "*_[0-9].tif"))
        self.img = torch.empty((self.aug_num * len(files), 3, 22, 17), dtype=torch.float32)
        self.label = torch.empty(len(files), dtype=torch.int64)
        for i, f in enumerate(files):
            self.img[self.aug_num * i:self.aug_num * i + self.aug_num] = TF.normalize(util.aug_by_translation(TF.to_tensor(Image.open(f)), self.aug_num, max_shift_len), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            self.label[i] = int(f[-5])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.img[idx], self.label[idx // self.aug_num]

    def __len__(self) -> int:
        return len(self.img)

class VidDataset(data.Dataset):
    def __init__(self, dir: str) -> None:
        files = glob(path.join(dir, "camera*/video_??-??-??_??.mkv"))
        self.cam_name = np.empty(len(files), dtype="<U3")
        self.vid_idx = np.empty(len(files), dtype=np.int32)
        self.img = torch.empty((len(files), 6, 3, 22, 17), dtype=torch.float32)
        self.label = np.empty(len(files), dtype=timedelta)
        for i, f in enumerate(files):
            self.cam_name[i] = path.basename(path.dirname(f))[6:]
            self.vid_idx[i] = int(f[-6:-4])
            for j, tmp_img in enumerate(util.extract_ts_fig(util.read_1st_frm(f))):
                self.img[i, j] = TF.normalize(TF.to_tensor(tmp_img), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            self.label[i] = util.calc_ts_from_name(f)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.img[idx // 6, idx % 6]

    def __len__(self) -> int:
        return 6 * len(self.label)

class DataModule(pl.LightningDataModule):
    def __init__(self, seed: int = 0, ts_fig_dir: Optional[str] = None, vid_dir: Optional[str] = None) -> None:
        super().__init__()

        self.dataset = {}
        self.seed = seed
        self.ts_fig_dir = ts_fig_dir
        self.vid_dir = vid_dir

    def setup(self, stage: str) -> None:
        match stage:
            case "fit" | "test":
                if self.ts_fig_dir is not None and "train" not in self.dataset.keys():
                    self.dataset["train"], self.dataset["validate"], self.dataset["test"] = data.random_split(TsFigDataset(self.ts_fig_dir), (0.8, 0.1, 0.1), generator=torch.Generator().manual_seed(self.seed))
            case "predict":
                self.dataset["predict"] = VidDataset(self.vid_dir)

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["train"], batch_size=256, shuffle=True, num_workers=4)

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["validate"], batch_size=256, num_workers=4)

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["test"], batch_size=256, num_workers=4)

    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["predict"], batch_size=6, num_workers=4)
