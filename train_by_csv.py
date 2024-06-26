import os.path as path
from glob import glob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.model as M
import script.utility as util
from script.data import DataModule4CsvAndTsFig


def train(csv_split_file: str, gpu_id: int, param_file: str, ts_fig_dirs: list[str], vid_dir: str, ckpt_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    param = util.load_param(param_file)
    param["enable_loss_weight"] = False
    model_cls = M.get_model_cls(param["arch"])

    trainer = pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_last=True, save_top_k=3),
        devices=[gpu_id],
        max_epochs=param["epoch"],
        accelerator="gpu"
    )
    datamodule = DataModule4CsvAndTsFig(csv_split_file, vid_dir, ts_fig_dirs, param, trainer.log_dir)

    if ckpt_file is None:
        trainer.fit(model_cls(param), datamodule=datamodule)
        ckpt_file = glob(path.join(trainer.log_dir, "checkpoints/", "epoch=*-step=*.ckpt"))[0]

    trainer.test(model=model_cls.load_from_checkpoint(ckpt_file), datamodule=datamodule)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--csv_split_file", required=True, help="specify csv split file", metavar="PATH_TO_CSV_SPLIT_FILE")
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-v", "--vid_dir", required=True, help="specify video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-d", "--ts_fig_dirs", nargs="*", default=[], help="specify list of timestamp figure dataset directories", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-c", "--ckpt_file", help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    train(args.csv_split_file, args.gpu_id, args.param_file, args.ts_fig_dirs, args.vid_dir, args.ckpt_file, args.result_dir_name)
