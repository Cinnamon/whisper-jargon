import sys
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
from whisper_finetune.model import WhisperModelModule

from whisper_main import whisper


def train():
    # load config
    config_path = Path("./config.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # dirs and paths
    out_log_dir = Path(config["path"]["log"])
    checkpoint_dir = Path(config["path"]["checkpoint"])
    with_timestamps = bool(config["data"]["timestamps"])
    device = "gpu" if torch.cuda.is_available() else "cpu"

    out_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # tools
    whisper_options = whisper.DecodingOptions(
        language=config["data"]["lang"], without_timestamps=not with_timestamps
    )
    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True, language=config["data"]["lang"], task=whisper_options.task
    )

    # logger
    tflogger = TensorBoardLogger(
        save_dir=out_log_dir, name=config["train_name"], version=config["train_id"]
    )

    csv_logger = CSVLogger(
        save_dir=out_log_dir, name=config["train_name"], version=config["train_id"]
    )
    # callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir / "checkpoint",
        filename="reazonspeechsmall-checkpoint-{epoch:04d}",
        save_top_k=-1,  # all model save
    )
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(
        config["train_manifest"],
        config["train_root"],
        config["val_manifest"],
        config["val_root"],
        config["train"],
        config["model_name"],
        config["data"]["lang"],
    )

    trainer = Trainer(
        precision=16,
        accelerator=device,
        max_epochs=config["train"]["num_train_epochs"],
        accumulate_grad_batches=config["train"]["gradient_accumulation_steps"],
        logger=csv_logger,
        callbacks=callback_list,
        num_nodes=1,
        devices=1,
        num_sanity_val_steps=2,
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
