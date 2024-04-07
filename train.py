import os.path as path
from glob import glob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.utility as util
from script.data import DataModule
from script.model import CNN3


def train(gpu_id: int, param: dict[str, util.Param] | str, ts_fig_dir: list[str], ckpt_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    if isinstance(param, str):
        param = util.load_param(param)
    param["enable_loss_weight"] = True

    datamodule = DataModule(param, ts_fig_dir)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_last=True),
        devices=[gpu_id],
        max_epochs=param["epoch"],
        accelerator="gpu"
    )

    if ckpt_file is None:
        datamodule.setup("fit")
        model = CNN3(param, datamodule.dataset["train"].calc_loss_weight())
        trainer.fit(model, datamodule=datamodule)
        model = CNN3.load_from_checkpoint(glob(path.join(trainer.log_dir, "checkpoints/", "epoch=*-step=*.ckpt"))[0], loss_weight=torch.empty(10, dtype=torch.float32))
    else:
        model = CNN3.load_from_checkpoint(ckpt_file, loss_weight=torch.empty(10, dtype=torch.float32))

    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ts_fig_dir", nargs="+", help="specify timestamp figure dataset directory", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")

    if sys.stdin.isatty():
        parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
        parser.add_argument("-c", "--ckpt_file", help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
        args = parser.parse_args()

        train(args.gpu_id, args.param_file, args.ts_fig_dir, args.ckpt_file, args.result_dir_name)

    else:
        args = parser.parse_args()
        lines = sys.stdin.readlines()

        train(args.gpu_id, json.loads(lines[1]), args.ts_fig_dir, lines[3].rstrip(), args.result_dir_name)
