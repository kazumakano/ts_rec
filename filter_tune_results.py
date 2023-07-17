import json
import math
import os.path as path
import pickle
from glob import glob, iglob
from os import listdir
import numpy as np
from tensorboard.backend.event_processing import tag_types
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import script.utility as util


def _match(cond: dict[str, util.Param], param: dict[str, util.Param]) -> bool:
    for k, v in cond.items():
        if v != param[k]:
            return False
    return True

def filter_tune_results(cond_file: str, result_dir: str) -> tuple[dict[str, util.Param], str, np.ndarray, list[tuple[dict[str, util.Param], float]]]:
    cond = util.load_param(cond_file)

    match_trials = []
    min_best_val_loss = math.inf
    for trial_dir in iglob(path.join(result_dir, "_try_*")):
        with open(path.join(trial_dir, "params.pkl"), mode="rb") as f:
            param = pickle.load(f)
        if "error.pkl" not in listdir(trial_dir) and _match(cond, param):
            accumulator = EventAccumulator(path.join(trial_dir, "log/version_0/"), size_guidance={tag_types.SCALARS: 0}).Reload()
            loss = np.array(([e.value for e in accumulator.Scalars("train_loss")], [e.value for e in accumulator.Scalars("validation_loss")]), dtype=np.float64).T
            best_val_loss = loss[:, 1].min()
            if best_val_loss < min_best_val_loss:
                min_ckpt_file = glob(path.join(trial_dir, "log/version_0/checkpoints/epoch=*-step=*.ckpt"))[0]
                min_loss = loss
                min_param = param
                min_best_val_loss = best_val_loss
            match_trials.append((param, best_val_loss))

    print("best parameters are")
    print(json.dumps(min_param))
    print("best model checkpoint file is")
    print(min_ckpt_file)

    return min_param, min_ckpt_file, min_loss, match_trials

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cond_file", required=True, help="specify condition file", metavar="PATH_TO_COND_FILE")
    parser.add_argument("-r", "--result_dir", required=True, help="specify experiment result directory", metavar="PATH_TO_RESULT_DIR")
    args = parser.parse_args()

    filter_tune_results(args.cond_file, args.result_dir)
