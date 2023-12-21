import pytorch_lightning as pl
import torch
from ray import train, tune
from ray.tune.experiment import Trial
from slack_bolt.app import App
from . import utility as util


class BestValLossReporter(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.best_val_loss = torch.inf

    def on_validation_epoch_end(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        if trainer.callback_metrics["validation_loss"] < self.best_val_loss:
            self.best_val_loss = trainer.callback_metrics["validation_loss"].item()

    def on_fit_end(self, _: pl.Trainer, __: pl.LightningModule) -> None:
        train.report({"best_validation_loss": self.best_val_loss})

class SlackBot(tune.Callback):
    def __init__(self, conf_file: str) -> None:
        super().__init__()
        self.conf = util.load_param(conf_file)
        self.client = App(signing_secret=self.conf["signing_secret"], token=self.conf["bot_token"]).client

    def on_trial_error(self, iteration: int, trials: list[Trial], trial: Trial) -> None:
        self.client.chat_postMessage(channel=self.conf["channel_id"], text=f"some error occurred in experiment {trial.experiment_dir_name}")

    def on_experiment_end(self, trials: list[Trial]) -> None:
        self.client.chat_postMessage(channel=self.conf["channel_id"], text=f"experiment {trials[0].experiment_dir_name} has finished")
