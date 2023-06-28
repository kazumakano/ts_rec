from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import script.utility as util
from script.data import DataModule
from script.model import CNN


def run(ckpt_file: str, gpu_id: int, param_file: str, vid_dir: str, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        devices=[gpu_id],
        accelerator="gpu"
    )

    trainer.predict(model=CNN.load_from_checkpoint(ckpt_file, param=util.load_param(param_file)), datamodule=DataModule(vid_dir=vid_dir))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt_file", required=True, help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
    parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
    parser.add_argument("-d", "--vid_dir", required=True, help="specify video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_file_name", help="specify result file name", metavar="RESULT_FILE_NAME")
    args = parser.parse_args()

    run(args.ckpt_file, args.gpu_id, args.param_file, args.vid_dir, args.result_file_name)
