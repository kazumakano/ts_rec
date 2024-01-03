import os.path as path
from datetime import datetime
from glob import iglob
import numpy as np
import pandas as pd


def _str2unix(ts_in_str: pd.Series) -> np.ndarray:
    ts_in_unix = np.empty(len(ts_in_str), dtype=np.float64)
    for i, t in ts_in_str.items():
        ts_in_unix[i] = datetime.strptime(t, "%H:%M:%S").timestamp()
    return ts_in_unix

def _unix2str(ts_in_unix: np.ndarray) -> np.ndarray:
    ts_in_str = np.empty(len(ts_in_unix), dtype="<U8")
    for i, t in enumerate(ts_in_unix):
        ts_in_str[i] = datetime.fromtimestamp(t).strftime("%H:%M:%S")
    return ts_in_str

def correct_predict_results(result_dir: str, ver: int = 0) -> None:
    for f in iglob(path.join(result_dir, f"*/??-??-??_*/version_{ver}/predict_results.csv")):
        results = pd.read_csv(f, usecols=("cam", "vid_idx", "frm_idx", "recog"))
        ts = _str2unix(results.loc[:, "recog"])

        ts_0 = ts - np.arange(0, 0.2 * len(ts) - 0.1, step=0.2)
        ts_0_median = np.median(ts_0)
        inlier_idxes = np.where((ts_0 >= ts_0_median - 1) & (ts_0 <= ts_0_median + 1))[0]
        results.loc[:, "recog"] = _unix2str(np.poly1d(np.polyfit(results.index[inlier_idxes], ts[inlier_idxes] + 0.5, 1))(results.index))
        results.to_csv(path.splitext(f)[0] + "_corrected.csv", index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_dir", required=True, help="specify predict result directory", metavar="PATH_TO_RESULT_DIR")
    parser.add_argument("-v", "--ver", default=0, type=int, help="specify version", metavar="VER")
    args = parser.parse_args()

    correct_predict_results(args.result_dir, args.ver)
