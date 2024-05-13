from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.model as M
import script.utility as util
from script.data import DataModule


def tune_weight(ckpt_file: str, gpu_id: int, param_file: str, ts_fig_dir: list[str], result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    param = util.load_param(param_file)
    param["enable_loss_weight"] = True
    model_cls = M.get_model_cls(param["arch"])

    datamodule = DataModule(param, ts_fig_dir, prop=(1, 0, 0))
    datamodule.val_files = datamodule.train_files
    datamodule.setup("fit")

    if torch.load(ckpt_file)["hyper_parameters"]["enable_loss_weight"]:
        model = model_cls.load_from_checkpoint(ckpt_file, param=param, loss_weight=datamodule.dataset["train"].calc_loss_weight())
    else:
        model = model_cls.load_from_checkpoint(ckpt_file, param=param)
        model.criterion.register_buffer("weight", datamodule.dataset["train"].calc_loss_weight())

    pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_top_k=-1, every_n_epochs=param["epoch"] // 10),
        devices=[gpu_id],
        max_epochs=param["epoch"],
        accelerator="gpu"
    ).fit(model, datamodule=datamodule)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt_file", required=True, help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-d", "--ts_fig_dir", nargs="+", help="specify timestamp figure dataset directory", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    tune_weight(args.ckpt_file, args.gpu_id, args.param_file, args.ts_fig_dir, args.result_dir_name)
