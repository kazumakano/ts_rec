import os.path as path
import random
import threading
from datetime import timedelta
from glob import glob, iglob
from os import makedirs
from queue import Queue
from typing import Generator, Literal, Optional, Self
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils import data
from torchvision.transforms import functional as TF
from tqdm import tqdm
from . import utility as util


class CsvDataset(data.Dataset):
    def __init__(self, csv_file: str, vid_dir: str, vid_idx: int, aug_num: int = 4, brightness: float = 0.2, contrast: float = 0.2, hue: float = 0.2, max_shift_len: int = 4, norm: bool = False, stride: int = 1) -> None:
        self.aug_num = aug_num

        df = pd.read_csv(csv_file, usecols=("cam", "vid_idx", "recog", "is_smudged"))
        df = df.loc[df.loc[:, "vid_idx"] == vid_idx]
        vid_files = glob(path.join(vid_dir, f"camera{df.iloc[0]['cam']}/video_??-??-??_{vid_idx:02d}.mp4"))
        if len(vid_files) != 1:
            raise Exception("video index must be unique")
        cap = cv2.VideoCapture(filename=vid_files[0])
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) != len(df):
            raise Exception("number of video frames and length of ground truth do not match")

        df = df.loc[::stride]
        self.img = torch.empty((6 * self.aug_num * len(df), 3, 22, 17), dtype=torch.float32)
        self.label = torch.empty(6 * len(df), dtype=torch.int64)
        sharp_frm_idxes = []
        for i, (_, r) in enumerate(tqdm(df.iterrows(), desc="loading timestamp figure images", total=len(df))):
            if pd.isna(r["is_smudged"]):
                for j, tmp_img in enumerate(util.extract_ts_fig(cap.read()[1])):
                    self.img[self.aug_num * (6 * i + j):self.aug_num * (6 * i + j + 1)] = util.aug_img(TF.to_tensor(tmp_img), self.aug_num, brightness, contrast, hue, max_shift_len)
                time_label = util.str2time(r["recog"])
                self.label[6 * i] = time_label.hour // 10
                self.label[6 * i + 1] = time_label.hour % 10
                self.label[6 * i + 2] = time_label.minute // 10
                self.label[6 * i + 3] = time_label.minute % 10
                self.label[6 * i + 4] = time_label.second // 10
                self.label[6 * i + 5] = time_label.second % 10
                sharp_frm_idxes.append(i)
            else:
                cap.read()

            for _ in range(stride - 1):
                cap.read()

        self.img = self.img[[self.aug_num * (6 * i + j) + k for i in sharp_frm_idxes for j in range(6) for k in range(self.aug_num)]]
        self.label = self.label[[6 * i + j for i in sharp_frm_idxes for j in range(6)]]

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

class TsFigDataset(data.Dataset):
    def __init__(self, files: list[str], aug_num: int = 64, brightness: float = 0.2, contrast: float = 0.2, hue: float = 0.2, max_shift_len: int = 4, norm: bool = False) -> None:
        self.aug_num = aug_num

        self.img = torch.empty((self.aug_num * len(files), 3, 22, 17), dtype=torch.float32)
        self.label = torch.empty(len(files), dtype=torch.int64)
        for i, f in enumerate(tqdm(files, desc="loading timestamp figure images")):
            self.img[self.aug_num * i:self.aug_num * i + self.aug_num] = util.aug_img(TF.to_tensor(cv2.imread(f)), self.aug_num, brightness, contrast, hue, max_shift_len)
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

    def calc_loss_weight(self) -> torch.Tensor:
        return torch.from_numpy(1 / (1 / self.breakdown).sum() / self.breakdown).to(dtype=torch.float32)

class VidDataset(data.Dataset):
    def __init__(self, files: list[str], frm_num: int = 5, norm: bool = False, sec_per_file: float = 1791, show_progress: bool = True) -> None:
        self.frm_num = frm_num

        self.cam_name = np.empty(len(files), dtype="<U3")
        self.vid_idx = np.empty(len(files), dtype=np.int32)
        self.img = torch.empty((self.frm_num * len(files), 6, 3, 22, 17), dtype=torch.float32)
        self.label = np.empty(len(files), dtype=timedelta)
        for i, f in enumerate(tqdm(files, desc="loading videos", disable=not show_progress)):
            self.cam_name[i] = path.basename(path.dirname(f))[6:]
            file_name = path.basename(f)
            self.vid_idx[i] = int(file_name[15:-4])
            for j, frm in enumerate(util.read_head_n_frms(f, self.frm_num)):
                for k, tmp_img in enumerate(util.extract_ts_fig(frm)):
                    self.img[self.frm_num * i + j, k] = TF.to_tensor(tmp_img)
                if norm:
                    self.img[self.frm_num * i + j] = TF.normalize(self.img[self.frm_num * i + j], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            self.label[i] = util.calc_ts_from_name(file_name, sec_per_file)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.img[idx // 6, idx % 6]

    def __len__(self) -> int:
        return 6 * len(self.img)

class VidDataset4Any(VidDataset):
    def __init__(self, file: str, max_frm_num: int, start_frm_idx: int = 0, norm: bool = False, show_progress: bool = True) -> None:
        self.start_frm_idx = start_frm_idx

        self.cam_name = path.basename(path.dirname(file))[6:]
        self.vid_idx = 0

        frms = util.read_head_n_frms(file, max_frm_num, True, self.start_frm_idx)
        self.img = torch.empty((len(frms), 6, 3, 22, 17), dtype=torch.float32)
        for i, frm in enumerate(tqdm(frms, desc="loading video", disable=not show_progress)):
            for j, tmp_img in enumerate(util.extract_ts_fig(frm)):
                self.img[i, j] = TF.to_tensor(tmp_img)
            if norm:
                self.img[i] = TF.normalize(self.img[i], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

class VidDataset4ManyFrms(VidDataset):
    def __init__(self, file: str, max_frm_num: int, label_at_start_frm: Optional[timedelta] = None, start_frm_idx: int = 0, norm: bool = False, sec_per_file: float = 1791, show_progress: bool = True) -> None:
        self.start_frm_idx = start_frm_idx

        self.cam_name = path.basename(path.dirname(file))[6:]
        file_name = path.basename(file)
        self.vid_idx = int(file_name[15:-4])
        self.label_at_start_frm = util.calc_ts_from_name(file_name, sec_per_file) if label_at_start_frm is None else label_at_start_frm

        frms = util.read_head_n_frms(file, max_frm_num, True, self.start_frm_idx)
        self.img = torch.empty((len(frms), 6, 3, 22, 17), dtype=torch.float32)
        for i, frm in enumerate(tqdm(frms, desc="loading video", disable=not show_progress)):
            for j, tmp_img in enumerate(util.extract_ts_fig(frm)):
                self.img[i, j] = TF.to_tensor(tmp_img)
            if norm:
                self.img[i] = TF.normalize(self.img[i], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

class DataModule(pl.LightningDataModule):
    def __init__(self, param: dict[str, util.Param], ts_fig_dirs: Optional[list[str]] = None, vid_dir: Optional[str] = None, ex_file: Optional[str] = None, prop: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> None:
        super().__init__()

        self.dataset: dict[str, TsFigDataset | VidDataset] = {}
        self.save_hyperparameters(param)

        if ts_fig_dirs is not None:
            files = []
            for d in ts_fig_dirs:
                files += glob(path.join(d, "*_[0-9].tif"))
            self.train_files, self.val_files, self.test_files = util.random_split(files, prop, seed)

        if vid_dir is not None:
            exclude = None if ex_file is None else util.load_param(ex_file)
            self.predict_files = []
            for d in sorted(iglob(path.join(vid_dir, "camera*"))):
                if exclude is None or exclude["camera"] is None or path.basename(d)[6:] not in exclude["camera"]:
                    for f in sorted(iglob(path.join(d, "video_??-??-??_*.mkv"))):
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
        return data.DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"], shuffle=self.hparams["shuffle"], num_workers=self.hparams["num_workers"], drop_last=self.hparams["drop_last"])

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["validate"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["predict"], batch_size=6, num_workers=self.hparams["num_workers"])

    @classmethod
    def load(cls, dir: str) -> Self:
        dataset, param = torch.load(path.join(dir, "data.pt"))
        self = cls(param)
        self.dataset = dataset

        return self

    def save(self, dir: str) -> None:
        torch.save((self.dataset, self.hparams), path.join(dir, "data.pt"))

    @staticmethod
    def unpack_param_list(param_list: dict[str, list[util.Param]]) -> dict[str, util.Param]:
        return {
            "batch_size": param_list["batch_size"][0],
            "num_workers": param_list["num_workers"][0],
            "shuffle": param_list["shuffle"][0]
        }

class _MultiDataLoader:
    def __init__(self, batch_size: int, drop_last: bool, shuffle: bool, num_workers: int, data_files: list[str], queue_size: int = 1) -> None:
        self.batch_size, self.drop_last, self.shuffle, self.num_workers, self.queue_size = batch_size, drop_last, shuffle, num_workers, queue_size
        self.data_files = data_files

    def __iter__(self) -> Generator[list[torch.Tensor], None, None]:
        data_queue = Queue(maxsize=self.queue_size)
        loader = threading.Thread(target=self._load_asyncly, args=(data_queue, ), daemon=True)
        loader.start()

        while loader.is_alive() or not data_queue.empty():
            for b in data.DataLoader(data_queue.get(), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=self.drop_last):
                yield b

    def _load_asyncly(self, queue: Queue) -> None:
        for f in random.sample(self.data_files, len(self.data_files)) if self.shuffle else self.data_files:
            queue.put(torch.load(f))

class DataModule4CsvAndTsFig(pl.LightningDataModule):
    def __init__(self, csv_split_file: str, vid_dir: str, ts_fig_dirs: list[str], param: dict[str, util.Param], result_dir: str, prop: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> None:
        random.seed(a=seed)
        super().__init__()

        self.data_files: dict[str, list[str]] = {}
        self.save_hyperparameters(param)
        self.result_dir = result_dir

        self.csv_split = util.load_param(csv_split_file)
        self.vid_dir = vid_dir

        files = []
        for d in ts_fig_dirs:
            files += glob(path.join(d, "*_[0-9].tif"))
        files = util.random_split(files, prop, seed)
        self.ts_fig_split = {"train": files[0], "validate": files[1], "test": files[2]}

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self._load_and_save("train")
                self._load_and_save("validate")
            case "test":
                self._load_and_save("test")

    def train_dataloader(self) -> data.DataLoader | _MultiDataLoader:
        if len(self.data_files["train"]) == 1:
            return data.DataLoader(torch.load(self.data_files["train"][0]), batch_size=self.hparams["batch_size"], shuffle=self.hparams["shuffle"], num_workers=self.hparams["num_workers"], drop_last=self.hparams["drop_last"])
        else:
            return _MultiDataLoader(self.hparams["batch_size"], self.hparams["drop_last"], self.hparams["shuffle"], self.hparams["num_workers"], self.data_files["train"], 1)

    def val_dataloader(self) -> data.DataLoader | _MultiDataLoader:
        if len(self.data_files["validate"]) == 1:
            return data.DataLoader(torch.load(self.data_files["validate"][0]), batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])
        else:
            return _MultiDataLoader(self.hparams["batch_size"], False, False, self.hparams["num_workers"], self.data_files["validate"], 1)

    def test_dataloader(self) -> data.DataLoader | _MultiDataLoader:
        if len(self.data_files["test"]) == 1:
            return data.DataLoader(torch.load(self.data_files["test"][0]), batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])
        else:
            return _MultiDataLoader(self.hparams["batch_size"], False, False, self.hparams["num_workers"], self.data_files["test"], 1)

    def _load_and_save(self, mode: Literal["train", "validate", "test"]) -> None:
        if not path.exists(self.result_dir):
            makedirs(self.result_dir)

        self.data_files[mode] = []
        i = 0
        for f in self.csv_split[mode]:
            for j in pd.read_csv(f, usecols=("vid_idx", )).loc[:, "vid_idx"].unique():
                dataset = CsvDataset(f, self.vid_dir, j) if mode == "train" else CsvDataset(f, self.vid_dir, j, 1)
                data_file = path.join(self.result_dir, f"{mode}_data_{i}.pt")
                torch.save(dataset, data_file)
                self.data_files[mode].append(data_file)
                i += 1
        dataset = TsFigDataset(self.ts_fig_split[mode]) if mode == "train" else TsFigDataset(self.ts_fig_split[mode], 1)
        data_file = path.join(self.result_dir, f"{mode}_data_{i}.pt")
        torch.save(dataset, data_file)
        self.data_files[mode].append(data_file)

class _DataLoaderMixer:
    def __init__(self, shuffle: bool, src: data.DataLoader | _MultiDataLoader, tgt: data.DataLoader) -> None:
        self.shuffle = shuffle
        self.src, self.tgt = self._use_eternal_loader(src), tgt

    def __iter__(self) -> Generator[list[torch.Tensor], None, None]:
        for tb in self.tgt:
            sb = next(self.src)
            batch = [torch.vstack((sb[0], tb[0])), torch.hstack((sb[1], tb[1]))]
            if self.shuffle:
                rand_idxes = torch.randperm(len(batch[1]))
                batch = [batch[0][rand_idxes], batch[1][rand_idxes]]
            yield batch

    def __len__(self) -> int:
        return len(self.tgt)

    @staticmethod
    def _use_eternal_loader(loader: data.DataLoader | _MultiDataLoader) -> Generator[list[torch.Tensor], None, None]:
        while True:
            for b in loader:
                yield b

class DataModuleMixer(pl.LightningDataModule):
    def __init__(self, param: dict[str, util.Param], src: DataModule | DataModule4CsvAndTsFig, tgt: DataModule, seed: int = 0) -> None:
        random.seed(a=seed)
        super().__init__()

        self.save_hyperparameters(param)
        self.src, self.tgt = src, tgt

    def setup(self, stage: str) -> None:
        self.src.setup(stage)
        self.tgt.setup(stage)

    def train_dataloader(self) -> _DataLoaderMixer:
        return _DataLoaderMixer(self.hparams["shuffle"], self.src.train_dataloader(), self.tgt.train_dataloader())

    def val_dataloader(self) -> data.DataLoader:
        return self.tgt.val_dataloader()
