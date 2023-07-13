import os
import os.path as path
import pickle
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import utils as tune_utils
from script.data import DataModule
import script.utility as util
from script.model import CNN3
from script.callback import BestValLossReporter, SlackBot

GPU_PER_TRIAL = 1
MAX_PEND_TRIAL_NUM = 3
VISIBLE_GPU = (1, 2, 3, )

def _get_grid_param_space(param_list: dict[str, list[util.Param]]) -> dict[str, dict[str, list[util.Param]]]:
    param_space = {}
    for k, l in param_list.items():
        param_space[k] = tune.grid_search(l)

    return param_space

def _try(param: dict[str, util.Param]) -> None:
    torch.set_float32_matmul_precision("high")
    # tune_utils.wait_for_gpu()

    if CNN3.is_valid_ks(param):
        datamodule = DataModule.load(path.join(tune.get_trial_dir(), "../"))
        trainer = pl.Trainer(
            logger=TensorBoardLogger(path.join(tune.get_trial_dir(), "log/"), name=None, default_hp_metric=False),
            callbacks=[BestValLossReporter(), ModelCheckpoint(monitor="validation_loss", save_last=True)],
            devices=1,
            enable_progress_bar=False,
            max_epochs=param["max_epoch"],
            accelerator="gpu",
            enable_model_summary=False
        )

        trainer.fit(CNN3(param, torch.from_numpy(len(datamodule.dataset["train"].label) / datamodule.dataset["train"].breakdown).to(dtype=torch.float32)), datamodule=datamodule)

def tune_params(param_list_file: str, ts_fig_dir: list[str], bot_conf_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in VISIBLE_GPU])
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(MAX_PEND_TRIAL_NUM)

    param_list = util.load_param(param_list_file)
    result_dir = util.get_result_dir(result_dir_name)

    datamodule = DataModule(util.unpack_param_list(param_list), ts_fig_dir)
    datamodule.setup("fit")
    datamodule.save(result_dir)

    tuner = tune.Tuner(
        trainable=tune.with_resources(_try, {"gpu": GPU_PER_TRIAL}),
        param_space=_get_grid_param_space(param_list),
        tune_config=tune.TuneConfig(mode="min", metric="best_validation_loss", chdir_to_trial_dir=False),
        run_config=air.RunConfig(name=path.basename(result_dir), storage_path=path.dirname(result_dir), callbacks=None if bot_conf_file is None else [SlackBot(bot_conf_file)])
    )

    results = tuner.fit()

    with open(path.join(result_dir, "tune_results.pkl"), mode="wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param_list_file", required=True, help="specify parameter list file", metavar="PATH_TO_PARAM_LIST_FILE")
    parser.add_argument("-d", "--ts_fig_dir", nargs="+", help="specify timestamp figure dataset directory", metavar="PATH_TO_TS_FIG_DIR")
    parser.add_argument("-b", "--bot_conf_file", help="enable slack bot", metavar="PATH_TO_BOT_CONF_FILE")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")
    args = parser.parse_args()

    tune_params(args.param_list_file, args.ts_fig_dir, args.bot_conf_file, args.result_dir_name)