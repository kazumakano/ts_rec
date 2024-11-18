import os.path as path
import pickle
from glob import iglob
import pandas as pd
import script.utility as util
from tqdm import tqdm

CAM_NAMES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")

def create(src_dir: str, tgt_file: str, ver: int) -> None:
    begin_dict, cache_dict = {}, {}
    for n in CAM_NAMES:
        results = []
        for f in sorted(iglob(path.join(src_dir, n, f"??-??-??_*/version_{ver}/interp_results.csv"))):
            results.append(pd.read_csv(f, usecols=("cam", "vid_idx", "frm_idx", "interp")))
        results = pd.concat(results)
        results.loc[:, "interp"] = results.loc[:, "interp"].map(util.str2sec)

        begin, end = results.iloc[:5]["interp"].max(), results.iloc[-5:]["interp"].min()
        cache = []
        for t in tqdm(range(begin, end + 1), desc=f"building cache for camera {n}"):
            results_of_t = results.loc[results.loc[:, "interp"] == t]
            for i in range(5):
                if i < len(results_of_t):
                    cache.append((int(results_of_t.iloc[i]["vid_idx"]), int(results_of_t.iloc[i]["frm_idx"])))
                else:
                    cache.append(cache[-1])
        begin_dict[n], cache_dict[n] = int(begin), cache

    with open(tgt_file, mode="wb") as f:
        pickle.dump((cache_dict, begin_dict), f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", required=True, help="specify source result directory", metavar="PATH_TO_SRC_DIR")
    parser.add_argument("-t", "--tgt_file", required=True, help="specify target cache file", metavar="PATH_TO_TGT_FILE")
    parser.add_argument("-v", "--ver", default=0, type=int, help="specify result version", metavar="VER")
    args = parser.parse_args()

    create(args.src_dir, args.tgt_file, args.ver)
