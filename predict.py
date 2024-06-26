import logging
import os.path as path
from datetime import datetime
from glob import iglob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
import script.model as M
import script.utility as util
from script.data import VidDataset


def predict(ckpt_file: str, gpu_id: int, param_file: str, vid_dir: str, ex_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    logging.disable()
    torch.set_float32_matmul_precision("high")

    param = util.load_param(param_file)
    result_dir = util.get_result_dir(result_dir_name)

    model = M.get_model_cls(param["arch"]).load_from_checkpoint(
        ckpt_file,
        map_location=torch.device("cuda", gpu_id),
        loss_weight=torch.empty(10, dtype=torch.float32) if param["enable_loss_weight"] else None
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu_id],
        logger=TensorBoardLogger(result_dir, name=None, default_hp_metric=False),
        enable_progress_bar=False
    )

    exclude = None if ex_file is None else util.load_param(ex_file)
    for d in tqdm(sorted(iglob(path.join(vid_dir, "camera*"))), desc="recognizing"):
        if exclude is None or exclude["camera"] is None or path.basename(d)[6:] not in exclude["camera"]:
            files = []
            for f in sorted(iglob(path.join(d, "video_??-??-??_*.mkv"))):
                if exclude is None or exclude["index"] is None or int(f[-6:-4]) not in exclude["index"]:
                    files.append(f)

            if len(files) > 0:
                trainer.predict(model=model, dataloaders=DataLoader(VidDataset(files, show_progress=False), batch_size=6, num_workers=param["num_workers"]))

    util.write_date(datetime.strptime(path.basename(path.normpath(vid_dir)), "%Y-%m-%d").date(), result_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt_file", required=True, help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-d", "--vid_dir", required=True, help="specify video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-e", "--ex_file", help="specify exclude file", metavar="PATH_TO_EX_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    predict(args.ckpt_file, args.gpu_id, args.param_file, args.vid_dir, args.ex_file, args.result_dir_name)
