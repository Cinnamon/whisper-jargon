import evaluate
import torch
from pytorch_lightning import LightningModule
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from whisper_main import whisper

from .dataset import WhisperASRDataCollator, WhisperASRDataset


class WhisperModelModule(LightningModule):
    def __init__(
        self,
        train_manifest,
        train_root,
        val_manifest,
        val_root,
        config,
        model_name="medium",
        lang="ja",
    ):
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name, device="cuda:3")
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=lang, task=self.options.task
        )

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.config = config
        self.train_root, self.train_manifest = train_root, train_manifest
        self.val_root, self.val_manifest = val_root, val_manifest
        self.train_dataset = WhisperASRDataset(
            self.train_manifest,
            self.train_root,
            self.tokenizer,
            sum_limit_duration=9999 * 3600,
        )
        self.val_dataset = WhisperASRDataset(
            self.val_manifest, self.val_root, self.tokenizer
        )
        # self.train_dataset = WhisperASRDataset(self.train_manifest_audio, self.train_manifest_text, self.train_root, self.tokenizer,
        #                                        sum_limit_duration=99 * 3600)
        # self.val_dataset = WhisperASRDataset(self.val_manifest_audio, self.val_manifest_text, self.val_root, self.tokenizer)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            # o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            # l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))

            o_str = self.tokenizer.decode(o)
            l_str = self.tokenizer.decode(l)
            for special_token in self.tokenizer.special_tokens:
                o_str = o_str.replace(special_token, "")
                l_str = l_str.replace(special_token, "")
            o_list.append(o_str)
            l_list.append(l_str)

        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {"cer": cer, "wer": wer, "loss": loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            eps=self.config["adam_epsilon"],
        )
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.t_total,
        )
        self.scheduler = scheduler

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.t_total = (
                (len(self.train_dataset) // (self.config["batch_size"]))
                // self.config["gradient_accumulation_steps"]
                * float(self.config["num_train_epochs"])
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            drop_last=True,
            shuffle=True,
            num_workers=self.config["num_worker"],
            collate_fn=WhisperASRDataCollator(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_worker"],
            collate_fn=WhisperASRDataCollator(),
        )
