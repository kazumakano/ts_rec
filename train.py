import os.path as path
from glob import glob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.model as M
import script.utility as util
from script.data import DataModule


def train(gpu_id: int, param_file: str, ts_fig_dirs: list[str], ckpt_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    param = util.load_param(param_file)
    param["enable_loss_weight"] = True
    model_cls = M.get_model_cls(param["arch"])

    datamodule = DataModule(param, ts_fig_dirs)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_last=True),
        devices=[gpu_id],
        max_epochs=param["epoch"],
        accelerator="gpu"
    )

    if ckpt_file is None:
        datamodule.setup("fit")
        trainer.fit(model_cls(param, datamodule.dataset["train"].calc_loss_weight()), datamodule=datamodule)
        ckpt_file = glob(path.join(trainer.log_dir, "checkpoints/", "epoch=*-step=*.ckpt"))[0]

    trainer.test(model=model_cls.load_from_checkpoint(ckpt_file, loss_weight=torch.empty(10, dtype=torch.float32)), datamodule=datamodule)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-d", "--ts_fig_dirs", nargs="+", required=True, help="specify list of timestamp figure dataset directories", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-c", "--ckpt_file", help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    train(args.gpu_id, args.param_file, args.ts_fig_dirs, args.ckpt_file, args.result_dir_name)
