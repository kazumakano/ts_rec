import logging
import os
import os.path as path
from datetime import datetime
from glob import iglob
from typing import Optional
import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
import script.model as M
import script.utility as util
from script.data import VidDataset4ManyFrms

"""
GPU_PER_TASK : float
    Number of gpus per one task.
MAX_FRM_NUM : int
    Maximum number of frames to load at once.
    Consider to reduce this if out of memory.
"""

GPU_PER_TASK = 1
MAX_FRM_NUM = 4096

@ray.remote(num_gpus=GPU_PER_TASK)
def _predict_by_file(ckpt_file: str, param: dict[str, util.Param], result_dir: str, vid_file: str) -> None:
    logging.disable()
    torch.set_float32_matmul_precision("high")

    model = M.get_model_cls(param["arch"], True).load_from_checkpoint(
        ckpt_file,
        map_location=torch.device("cuda", 0),
        loss_weight=torch.empty(10, dtype=torch.float32) if param["enable_loss_weight"] else None
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(result_dir, name=None, default_hp_metric=False),
        enable_progress_bar=False
    )

    dataset_idx = 0
    while True:
        dataset = VidDataset4ManyFrms(vid_file, MAX_FRM_NUM, None if dataset_idx == 0 else model.ts_at_end_frm, dataset_idx * MAX_FRM_NUM, show_progress=False)
        if len(dataset) == 0:
            break
        trainer.predict(model=model, dataloaders=DataLoader(dataset, batch_size=param["batch_size"], num_workers=param["num_workers"]))
        dataset_idx += 1

def predict_all_frms(ckpt_file: str, param: dict[str, util.Param] | str, vid_dir: str, ex_file: Optional[str] = None, gpu_ids: Optional[list[int]] = None, result_dir_name: Optional[str] = None) -> None:
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids])
    ray.init()

    if isinstance(param, str):
        param = util.load_param(param)
    if param["batch_size"] % 6 != 0:
        raise Exception("batch size must be divisible by 6")
    result_dir = util.get_result_dir(result_dir_name)

    exclude = None if ex_file is None else util.load_param(ex_file)
    pid_queue = []
    for d in sorted(iglob(path.join(vid_dir, "camera*"))):
        cam_name = path.basename(d)[6:]
        if exclude is None or exclude["camera"] is None or cam_name not in exclude["camera"]:
            for f in tqdm(sorted(iglob(path.join(d, "video_??-??-??_*.mp4"))), desc=f"recognizing camera {cam_name}"):
                if exclude is None or exclude["index"] is None or int(f[-6:-4]) not in exclude["index"]:
                    if len(pid_queue) >= cuda.device_count() // GPU_PER_TASK:
                        pid_queue.remove(ray.wait(pid_queue, num_returns=1)[0][0])
                    pid_queue.append(_predict_by_file.remote(path.abspath(ckpt_file), param, path.join(result_dir, cam_name, path.splitext(path.basename(f))[0][6:]), path.abspath(f)))

    ray.get(pid_queue)

    util.write_date(datetime.strptime(path.basename(path.normpath(vid_dir)), "%Y-%m-%d").date(), result_dir)

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--vid_dir", required=True, help="specify video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-e", "--ex_file", help="specify exclude file", metavar="PATH_TO_EX_FILE")
    parser.add_argument("-g", "--gpu_ids", nargs="*", type=int, help="specify list of GPU device IDs", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")

    if sys.stdin.isatty():
        parser.add_argument("-c", "--ckpt_file", required=True, help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
        parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
        args = parser.parse_args()

        predict_all_frms(args.ckpt_file, args.param_file, args.vid_dir, args.ex_file, args.gpu_ids, args.result_dir_name)

    else:
        args = parser.parse_args()
        lines = sys.stdin.readlines()

        predict_all_frms(lines[3].rstrip(), json.loads(lines[1]), args.vid_dir, args.ex_file, args.gpu_ids, args.result_dir_name)
