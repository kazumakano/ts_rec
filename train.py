import os.path as path
from glob import glob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.utility as util
from script.data import DataModule
from script.model import CNN, CNNDeep


def train(gpu_id: int, param_file: str, ts_fig_dir: str, ckpt_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    param = util.load_param(param_file)

    datamodule = DataModule(param, ts_fig_dir)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_last=True),
        devices=[gpu_id],
        max_epochs=param["max_epoch"],
        accelerator="gpu"
    )

    if ckpt_file is None:
        datamodule.setup("fit")
        model = CNNDeep(param, torch.Tensor([len(datamodule.dataset["train"]) / v for v in datamodule.dataset["train"].breakdown.values()]))
        trainer.fit(model, datamodule=datamodule)
        model.load_from_checkpoint(glob(path.join(trainer.log_dir, "checkpoints/", "epoch=*-step=*.ckpt"))[0], loss_weight=torch.empty(10, dtype=torch.float32))
    else:
        model = CNNDeep.load_from_checkpoint(ckpt_file, param=param, loss_weight=torch.empty(10, dtype=torch.float32))

    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-d", "--ts_fig_dir", nargs="+", help="specify timestamp figure dataset directory", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-c", "--ckpt_file", help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    train(args.gpu_id, args.param_file, args.ts_fig_dir, args.ckpt_file, args.result_dir_name)
