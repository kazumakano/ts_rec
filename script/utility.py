import csv
import math
import os.path as path
import pickle
from datetime import date, datetime, time, timedelta
from glob import glob, iglob
from typing import Optional, overload
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
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

def calc_ts_from_name(file_name: str, sec_per_file: float) -> timedelta:
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

    return timedelta(seconds=int(file_name[12:14]) + sec_per_file * int(file_name[15:-4]), minutes=int(file_name[9:11]), hours=int(file_name[6:8]))

def check_ts_consis(ts: np.ndarray, label_at_start_frm: Optional[timedelta] = None) -> list[int]:
    """
    Check consistency of sequential timestamps.

    Parameters
    ----------
    ts : ndarray[timedelta]
        Sequential timestamps.
        Shape is (frame, ).
    label_at_start_frm : timedelta, optional
        Timestamp at start frame.

    Returns
    -------
    frm_idxes : list[int]
        Frame indexes when timestamps are inconsistent.
    """

    inconsis_frm_idxes = []
    for i, t in enumerate(ts):
        if (i == 0 and label_at_start_frm is not None and (t - label_at_start_frm).total_seconds() not in (0, 1)) or (i > 0 and (t - ts[i - 1]).total_seconds()) not in (0, 1):
            inconsis_frm_idxes.append(i)

    return inconsis_frm_idxes

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

def get_consis_ts(estim: np.ndarray, label_at_start_frm: timedelta, label_is_accurate: bool, max_reliable_order: int = 2) -> tuple[np.ndarray, list[int]]:
    """
    Decide timestamps based on sequential model outputs and last one, considering time consistency.

    Parameters
    ----------
    estim : ndarray[float32]
        Sequential model outputs.
        Shape is (frame, 6, class).
    label_at_start_frm : timedelta
        Timestamp at start frame.
    label_is_accurate : bool
        Whether label at start frame is accurate.
    max_reliable_order : int, optional
        Maximum order of estimation probability to allow.
        Must be 0 ~ 9.

    Returns
    -------
    ts : ndarray[timedelta]
        Likely timestamps.
        Shape is (frame, ).
    frm_idxes : list[int]
        Frame indexes when timestamps are inconsistent.
    """

    likely_ts = np.empty(len(estim), dtype=timedelta)
    inconsis_frm_idxes = []
    for i, (h_1, h_2, m_1, m_2, s_1, s_2) in enumerate(softmax(-estim, axis=2).argsort(axis=2)):
        if i == 0 and not label_is_accurate:
            label_in_sec = round(label_at_start_frm.total_seconds())

            if label_in_sec % 3600 >= 3540:    # label is ??:59:??
                m_1_ordrer_if_same, m_2_order_if_same = np.where(m_1[m_1 < 6] == 5)[0][0], np.where(m_2 == 9)[0][0]
                m_1_ordrer_if_next, m_2_order_if_next = np.where(m_1[m_1 < 6] == 0)[0][0], np.where(m_2 == 0)[0][0]
                if m_1_ordrer_if_same + m_2_order_if_same < m_1_ordrer_if_next + m_2_order_if_next:
                    likely_ts[i] = timedelta(seconds=10 * round(s_1[s_1 < 6][0]) + round(s_2[0]), minutes=59, hours=label_in_sec // 3600)
                else:
                    likely_ts[i] = timedelta(seconds=10 * round(s_1[s_1 < 6][0]) + round(s_2[0]), minutes=0, hours=label_in_sec // 3600 + 1)
            elif label_in_sec % 600 >= 540:    # label is ??:[0-4]9:??
                m_1_ordrer_if_same, m_2_order_if_same = np.where(m_1[m_1 < 6] == label_in_sec % 3600 // 600)[0][0], np.where(m_2 == 9)[0][0]
                m_1_ordrer_if_next, m_2_order_if_next = np.where(m_1[m_1 < 6] == label_in_sec % 3600 // 600 + 1)[0][0], np.where(m_2 == 0)[0][0]
                if m_1_ordrer_if_same + m_2_order_if_same < m_1_ordrer_if_next + m_2_order_if_next:
                    likely_ts[i] = timedelta(seconds=10 * round(s_1[s_1 < 6][0]) + round(s_2[0]), minutes=label_in_sec % 3600 // 60, hours=label_in_sec // 3600)
                else:
                    likely_ts[i] = timedelta(seconds=10 * round(s_1[s_1 < 6][0]) + round(s_2[0]), minutes=label_in_sec % 3600 // 60 + 1, hours=label_in_sec // 3600)
            else:                              # label is ??:?[0-8]:??
                m_2_order = 0
                while True:
                    if m_2_order > max_reliable_order:
                        likely_ts[i] = timedelta(seconds=10 * round(s_1[s_1 < 6][0]) + round(s_2[0]), minutes=10 * round(m_1[m_1 < 6][0]) + round(m_2[0]), hours=10 * round(h_1[h_1 < 3][0]) + round(h_2[0]))
                        inconsis_frm_idxes.append(0)
                        break
                    elif round(m_2[m_2_order]) - label_in_sec % 600 // 60 in (0, 1):
                        likely_ts[i] = timedelta(seconds=10 * round(s_1[s_1 < 6][0]) + round(s_2[0]), minutes=10 * (label_in_sec % 3600 // 600) + round(m_2[m_2_order]), hours=label_in_sec // 3600)
                        break
                    else:
                        m_2_order += 1

        else:
            label = label_at_start_frm if i == 0 else likely_ts[i - 1]
            label_in_sec = round(label.total_seconds())

            if label_in_sec % 60 == 59:       # label is ??:??:59
                s_1_order_if_same, s_2_order_if_same = np.where(s_1[s_1 < 6] == 5)[0][0], np.where(s_2 == 9)[0][0]
                s_1_order_if_next, s_2_order_if_next = np.where(s_1[s_1 < 6] == 0)[0][0], np.where(s_2 == 0)[0][0]
                if s_1_order_if_same + s_2_order_if_same < s_1_order_if_next + s_2_order_if_next:
                    likely_ts[i] = label
                else:
                    likely_ts[i] = label + timedelta(seconds=1)
            else:
                if label_in_sec % 10 == 9:    # label is ??:??:[0-4]9
                    s_1_order_if_same, s_2_order_if_same = np.where(s_1[s_1 < 6] == label_in_sec % 60 // 10)[0][0], np.where(s_2 == 9)[0][0]
                    s_1_order_if_next, s_2_order_if_next = np.where(s_1[s_1 < 6] == label_in_sec % 60 // 10 + 1)[0][0], np.where(s_2 == 0)[0][0]
                    if s_1_order_if_same + s_2_order_if_same < s_1_order_if_next + s_2_order_if_next:
                        likely_ts[i] = label
                    else:
                        likely_ts[i] = label + timedelta(seconds=1)
                else:                         # label is ??:??:?[0-8]
                    s_2_order = 0
                    while True:
                        if s_2_order > max_reliable_order:
                            likely_ts[i] = timedelta(seconds=10 * round(s_1[s_1 < 6][0]) + round(s_2[0]), minutes=10 * round(m_1[m_1 < 6][0]) + round(m_2[0]), hours=10 * round(h_1[h_1 < 3][0]) + round(h_2[0]))
                            inconsis_frm_idxes.append(i)
                            break
                        else:
                            diff_in_sec = round(s_2[s_2_order]) - label_in_sec % 10
                            if diff_in_sec in (0, 1):
                                likely_ts[i] = label + timedelta(seconds=diff_in_sec)
                                break
                            else:
                                s_2_order += 1

    return likely_ts, inconsis_frm_idxes

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

def load_test_result(result_dir: str, ver: int = 0) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Param]]:
    with open(path.join(result_dir, f"version_{ver}/", "test_outputs.pkl"), mode="rb") as f:
        return pickle.load(f), load_param(path.join(result_dir, f"version_{ver}/", "hparams.yaml"))

def plot_all_predict_results(result_dir: str, ver: int = 0) -> None:
    with open(path.join(result_dir, "date.txt")) as f:
        true_date = datetime.strptime(f.readline(), "%Y-%m-%d\n").date()

    dirs = glob(path.join(result_dir, "*/"))
    fig, axes = plt.subplots(nrows=math.ceil(len(dirs) / 2), ncols=2, figsize=(16, 4 * math.ceil(len(dirs) / 2)))
    for i, d in enumerate(dirs):
        idx = np.empty(0, dtype=np.int64)
        ts = np.empty(0, dtype=datetime)
        for f in iglob(path.join(d, f"??-??-??_*/version_{ver}/predict_results.csv")):
            results = pd.read_csv(f, usecols=("recog", ))
            idx = np.hstack((idx, results.index + (0 if len(idx) == 0 else idx[-1])))
            tmp = np.empty(len(results), dtype=datetime)
            for j, t in enumerate(results.loc[:, "recog"]):
                tmp[j] = datetime.strptime(t, "%H:%M:%S") + (true_date - date(1900, 1, 1))
            ts = np.hstack((ts, tmp))

        axes[i // 2, i % 2].scatter(idx, ts, s=1)
        axes[i // 2, i % 2].set_title(f"camera {path.basename(path.normpath(d))}")
    fig.show()

def plot_predict_results_by_cam(cam_name: str, result_dir: str, ver: int = 0) -> None:
    with open(path.join(result_dir, "date.txt")) as f:
        true_date = datetime.strptime(f.readline(), "%Y-%m-%d\n").date()

    files = glob(path.join(result_dir, cam_name, f"??-??-??_*/version_{ver}/predict_results.csv"))
    fig, axes = plt.subplots(nrows=math.ceil(len(files) / 2), ncols=2, figsize=(16, 4 * math.ceil(len(files) / 2)))
    for i, f in enumerate(files):
        results = pd.read_csv(f, usecols=("recog", ))
        ts = np.empty(len(results), dtype=datetime)
        for j, t in enumerate(results.loc[:, "recog"]):
            ts[j] = datetime.strptime(t, "%H:%M:%S") + (true_date - date(1900, 1, 1))

        axes[i // 2, i % 2].scatter(results.index, ts, s=1)
        axes[i // 2, i % 2].set_title(f"video {path.basename(path.dirname(path.dirname(f)))[9:]}")
    fig.show()

def plot_breakdown(breakdown: np.ndarray) -> None:
    plt.pie(breakdown, labels=range(10), autopct="%d%%", startangle=90, counterclock=False)
    plt.show()

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

def read_head_n_frms(file: str, n: int, skip_one_by_one: bool = False, start_idx: int = 0) -> np.ndarray:
    """
    Read first n frames.

    Parameters
    ----------
    file : str
        Path to video file.
    n : int
        Maximum number of frames to read.
    skip_one_by_one : bool, optional
        Whether to skip frames one by one.
    start_idx : int, optional
        Index of first frame to read.

    Returns
    -------
    frms : ndarray[uint8]
        Frame images.
        Shape is (n, height, width, channel).
    """

    cap = cv2.VideoCapture(filename=file)
    if skip_one_by_one:
        for _ in range(start_idx):
            cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frms = []
    for _ in range(n):
        ret, frm = cap.read()
        if not ret:
            break
        frms.append(frm)

    return np.empty((0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8) if len(frms) == 0 else np.stack(frms)

def str2time(ts: str) -> time:
    return datetime.strptime(ts, "%H:%M:%S").time()

def timedelta2str(ts: timedelta, restrict_fmt: bool = True) -> str:
    """
    Convert timestamp to string.

    Parameters
    ----------
    ts : timedelta
        Timestamp in timedelta.
    restrict_fmt : bool, optional
        Whether to restrict hour to no more than 24.

    Returns
    -------
    ts : str
        Timestamp in string.

    Examples
    --------
    >>> timedelta2str(timedelta(days=1, hours=1, minutes=23, seconds=45))
    '01:23:45'
    >>> timedelta2str(timedelta(days=1, hours=1, minutes=23, seconds=45), restrict_fmt=False)
    '25:23:45'
    """

    return f"{(ts % timedelta(days=1) if restrict_fmt else ts) // timedelta(hours=1):02d}:{ts % timedelta(hours=1) // timedelta(minutes=1):02d}:{ts % timedelta(minutes=1) // timedelta(seconds=1):02d}"

def write_date(date: date, result_dir: str) -> None:
    with open(path.join(result_dir, "date.txt"), mode="w") as f:
        f.write(str(date) + "\n")

@overload
def write_predict_result(cam_name: np.ndarray, vid_idx: np.ndarray, ts: np.ndarray, label: np.ndarray, frm_num: int, result_dir: str) -> None:
    ...

@overload
def write_predict_result(cam_name: str, vid_idx: int, ts: np.ndarray, label_at_start_frm: timedelta, start_frm_idx: int, result_dir: str, inconsis_frm_idxes: list[int]) -> None:
    ...

def write_predict_result(cam_name: np.ndarray | str, vid_idx: np.ndarray | int, ts: np.ndarray, label: np.ndarray | timedelta, frm_num_or_start_frm_idx: int, result_dir: str, inconsis_frm_idxes: list[int] | None = None) -> None:
    with open(path.join(result_dir, "predict_results.csv"), mode="a") as f:
        writer = csv.writer(f)

        if isinstance(cam_name, str):
            if f.tell() == 0:
                writer.writerow(("cam", "vid_idx", "frm_idx", "recog", "diff_in_sec", "is_inconsis"))
            t: timedelta
            for i, t in enumerate(ts):
                writer.writerow((cam_name, vid_idx, frm_num_or_start_frm_idx + i, timedelta2str(t), (t.total_seconds() - label.total_seconds() + 43200) % 86400 - 43200, "inconsis" if i in inconsis_frm_idxes else None))
        else:
            if f.tell() == 0:
                writer.writerow(("cam", "vid_idx", "frm_idx", "recog", "diff_in_sec"))
            t: timedelta
            for i, t in enumerate(ts):
                writer.writerow((cam_name[i // frm_num_or_start_frm_idx], vid_idx[i // frm_num_or_start_frm_idx], i % frm_num_or_start_frm_idx, timedelta2str(t), (t.total_seconds() - label[i // frm_num_or_start_frm_idx].total_seconds() + 43200) % 86400 - 43200))
