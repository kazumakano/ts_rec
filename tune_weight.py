import io
from typing import Optional
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.model as M
import script.utility as util
from script.data import DataModule, DataModule4CsvAndTsFig, DataModuleMixer


def tune_weight(ckpt_file: str, gpu_id: int, param_file: str, src_csv_split_file: str, src_vid_dir: str, src_ts_fig_dirs: list[str], tgt_ts_fig_dirs: list[str], prop: list[float], result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    param = util.load_param(param_file)
    param["enable_loss_weight"] = False
    model_cls = M.get_model_cls(param["arch"])

    if torch.load(ckpt_file)["hyper_parameters"]["enable_loss_weight"]:
        model = model_cls.load_from_checkpoint(ckpt_file, param=param, loss_weight=torch.empty(10, dtype=torch.float32))
        model.criterion.register_buffer("weight", None)
    else:
        model = model_cls.load_from_checkpoint(ckpt_file, param=param)

    tgt_datamodule = DataModule({**param, "batch_size": round(prop[1] * param["batch_size"])}, tgt_ts_fig_dirs, prop=(1, 0, 0))
    tgt_datamodule.val_files = tgt_datamodule.train_files

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu_id],
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(save_top_k=-1, every_n_epochs=param["epoch"] // 10),
        max_epochs=param["epoch"]
    )
    trainer.fit(
        model,
        datamodule=DataModuleMixer(
            param,
            DataModule4CsvAndTsFig(io.StringIO(initial_value=yaml.safe_dump({"train": sum(util.load_param(src_csv_split_file).values(), start=[]), "validate": []})), src_vid_dir, src_ts_fig_dirs, {**param, "batch_size": round(prop[0] * param["batch_size"])}, trainer.log_dir, (1, 0, 0)),
            tgt_datamodule
        )
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt_file", required=True, help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-ss", "--src_csv_split_file", required=True, help="specify source csv split file", metavar="PATH_TO_CSV_SPLIT_FILE")
    parser.add_argument("-sv", "--src_vid_dir", required=True, help="specify source video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-sd", "--src_ts_fig_dirs", nargs="*", default=[], help="specify list of source timestamp figure dataset directories", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-td", "--tgt_ts_fig_dirs", nargs="+", required=True, help="specify list of target timestamp figure dataset directories", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("--prop", nargs=2, default=(0.99, 0.01), type=float, help="specify mix proportion", metavar="PROP")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    tune_weight(args.ckpt_file, args.gpu_id, args.param_file, args.src_csv_split_file, args.src_vid_dir, args.src_ts_fig_dirs, args.tgt_ts_fig_dirs, args.prop, args.result_dir_name)
