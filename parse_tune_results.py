import csv
import json
import os.path as path
import pickle
from glob import iglob
from os import listdir
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from matplotlib import pyplot as plt
from ray import air
from tensorboard.backend.event_processing import tag_types
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import script.utility as util


def parse_tune_results(result_dir: str, export_errs: bool = False, export_loss: bool = False) -> tuple[dict[str, util.Param], Path, np.ndarray, list[tuple[dict[str, util.Param], Any]]]:
    with open(path.join(result_dir, "tune_results.pkl"), mode="rb") as f:
        best_result: air.Result = pickle.load(f).get_best_result()

    print("best parameters are")
    print(json.dumps(best_result.config))

    ckpt_file = tuple(best_result.log_dir.joinpath("log/version_0/checkpoints/").glob("epoch=*-step=*.ckpt"))[0]
    print("best model checkpoint file is")
    print(ckpt_file)

    accumulator = EventAccumulator(best_result.log_dir.joinpath("log/version_0/").as_posix(), size_guidance={tag_types.SCALARS: 0}).Reload()
    loss = np.array(([e.value for e in accumulator.Scalars("train_loss")], [e.value for e in accumulator.Scalars("validation_loss")]), dtype=np.float64).T
    if export_loss:
        with open(path.join(result_dir, "loss.csv"), mode="w") as f:
            csv.writer(f).writerows(loss)
        plt.plot(loss[:, 0])
        plt.plot(loss[:, 1])
        plt.savefig(path.join(result_dir, "loss.png"))
        plt.close()
        print("exported to loss.csv and loss.png")

    errs = []
    for trial_dir in iglob(path.join(result_dir, "_try_*")):
        if "error.pkl" in listdir(trial_dir):
            with open(path.join(trial_dir, "error.pkl"), mode="rb") as f:
                cause = pickle.load(f).cause
            with open(path.join(trial_dir, "params.pkl"), mode="rb") as f:
                param = pickle.load(f)
            errs.append((param, cause))
    if export_errs:
        with open(path.join(result_dir, "errors.txt"), mode="w") as f:
            for p, c in errs:
                f.write(yaml.dump(p) + "\n")
                f.write(str(c) + "\n\n")
        print("exported to errors.txt")

    return best_result.config, ckpt_file, loss, errs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_dir", required=True, help="specify experiment result directory", metavar="PATH_TO_RESULT_DIR")
    parser.add_argument("-e", "--err", action="store_true", help="export errors")
    parser.add_argument("-l", "--loss", action="store_true", help="export loss as csv and image")
    args = parser.parse_args()

    parse_tune_results(args.result_dir, args.err, args.loss)
