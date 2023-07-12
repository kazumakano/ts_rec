import csv
import os.path as path
from datetime import datetime, timedelta
import cv2
import numpy as np
import torch
import yaml
from scipy.special import softmax
from torchvision import transforms as T


Param = bool | float | int

def aug_img(img: torch.Tensor, aug_num: int, brightness: float, contrast: float, hue: float, max_shift_len: int) -> torch.Tensor:
    """
    Augment image by randomly coloring and translation.

    Parameters
    ----------
    img : Tensor[float32]
        Original image.
        Shape is (channel, height, width).
    aug_num : int
        The number of images to augment.
    brightness : float
        How much to jitter brightness.
    contrast : float
        How much to jitter contrast.
    hue : float
        How much to jitter hue.
    max_shift_len : int
        Maximum length to shift image.

    Returns
    -------
    imgs : Tensor[float32]
        Augmented images.
        Shape is (aug_num, channel, height, width).
    """

    jitter_color = T.ColorJitter(brightness=brightness, contrast=contrast, hue=hue)
    translate = T.RandomAffine(0, translate=(max_shift_len / img.shape[2], max_shift_len / img.shape[1]))

    auged_imgs = torch.empty((aug_num, *img.shape), dtype=torch.float32)
    auged_imgs[0] = img
    for i in range(1, aug_num):
        auged_imgs[i] = translate(jitter_color(img))

    return auged_imgs

def calc_ts_from_name(file: str, sec_per_file: float) -> timedelta:
    """
    Roughly calculate timestamp based on video file name.

    Parameters
    ----------
    file : str
        Path to video file.
    sec_per_file : float
        Typical video length [s].

    Returns
    ------
    ts : timedelta
        Timestamp at the start of video.
    """

    return timedelta(seconds=int(file[-9:-7]) + sec_per_file * int(file[-6:-4]), minutes=int(file[-12:-10]), hours=int(file[-15:-13]))

def extract_ts_fig(frm: np.ndarray) -> np.ndarray:
    """
    Extract images of timestamp figures from video frame.

    Parameters
    ----------
    frm : ndarray[uint8]
        Frame image.
        Shape is (height, width, channel).

    Returns
    -------
    imgs : ndarray[uint8]
        Images of every timestamp figure.
        Shape is (6, height, width, channel).
    """

    ts_fig_imgs = np.empty((6, 22, 17, 3), dtype=np.uint8)
    for i, digit in enumerate((0, 1, 3, 4, 6, 7)):
        ts_fig_imgs[i] = frm[19:41, 18 * digit + 198:18 * digit + 215]

    return ts_fig_imgs

def get_most_likely_ts(estim: np.ndarray) -> np.ndarray:
    """
    Decide most likely timestamps based on model outputs, eliminating invalid timestamps.

    Parameters
    ----------
    estim : ndarray[float32]
        Model outputs.
        Shape is (batch, 6, class).

    Returns
    -------
    ts : ndarray[timedelta]
        Most likely timestamps.
        Shape is (batch, ).
    """

    most_likely_ts = np.empty(len(estim), dtype=timedelta)
    for i, (h_1, h_2, m_1, m_2, s_1, s_2) in enumerate(softmax(-estim, axis=2).argsort(axis=2)):
        most_likely_ts[i] = timedelta(seconds=10 * int(s_1[s_1 < 6][0]) + int(s_2[0]), minutes=10 * int(m_1[m_1 < 6][0]) + int(m_2[0]), hours=10 * int(h_1[h_1 < 3][0]) + int(h_2[0]))

    return most_likely_ts

def get_result_dir(dir_name: str | None) -> str:
    if dir_name is None:
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return path.join(path.dirname(__file__), "../result/", dir_name)

def load_param(file: str) -> dict[str, Param | list[Param] | list[str]]:
    with open(file) as f:
        return yaml.safe_load(f)

def random_split(files: list[str], prop: tuple[float, float, float], seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    mixed_idxes = torch.randperm(len(files), generator=torch.Generator().manual_seed(seed), dtype=torch.int32).numpy()

    train_num = round(prop[0] * len(mixed_idxes) / sum(prop))
    train_files = []
    for i in mixed_idxes[:train_num]:
        train_files.append(files[i])

    val_num = round(prop[1] * len(mixed_idxes) / sum(prop))
    val_files = []
    for i in mixed_idxes[train_num:train_num + val_num]:
        val_files.append(files[i])

    test_files = []
    for i in mixed_idxes[train_num + val_num:]:
        test_files.append(files[i])

    return train_files, val_files, test_files

def read_head_n_frms(file: str, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(filename=file)
    frms = []
    for _ in range(n):
        ret, frm = cap.read()
        if not ret:
            break
        frms.append(frm)

    return np.stack(frms)

def unpack_param_list(param_list: dict[str, list[Param]]) -> dict[str, Param]:
    param = {}
    for k, v in param_list.items():
        param[k] = v[0]

    return param

def write_predict_result(cam_name: np.ndarray, vid_idx: np.ndarray, ts: np.ndarray, label: np.ndarray, result_dir: str) -> None:
    with open(path.join(result_dir, "predict_results.csv"), mode="a") as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow(("cam", "idx", "recog", "diff_in_sec"))
        for i in range(len(cam_name)):
            writer.writerow((cam_name[i], vid_idx[i], str(ts[i]), ts[i].total_seconds() - label[i].total_seconds()))
