import csv
import json
import os
from pathlib import Path
from typing import Union

import ffmpeg
import numpy as np
import torch
import tqdm

from whisper_main.whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper_main.whisper.tokenizer import Tokenizer


class WhisperASRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        f_manifest: str,
        dir_root: str,
        tokenizer: Tokenizer,
        max_duration: float = 30,
        min_duration: float = 0.1,
        sum_limit_duration: float = -1,
    ):

        super().__init__()
        self.tokenizer = tokenizer
        self.manifest = []

        self.dir_root, self.f_manifest = dir_root, f_manifest
        sum_duration = 0
        with open(f_manifest, newline="") as f_csv:
            reader = csv.DictReader(f_csv, restkey="unknown", restval="-1")
            for idx, row in enumerate(tqdm.tqdm(reader)):
                key = row["key"]
                f_audio = row["audio"]
                f_info = row["info"]
                with open(os.path.join(dir_root, f_info), "r") as f:
                    info = json.load(f)

                duration = info["duration"]
                if (max_duration and duration > max_duration) or (
                    min_duration and duration < min_duration
                ):
                    continue

                transcript = info["transcript"].replace("\n", "").replace("\r", "")
                if len(transcript) == 0:
                    continue
                self.manifest.append(
                    {
                        "key": key,
                        "audio": os.path.join(dir_root, f_audio),
                        "duration": duration,
                        "transcript": transcript,
                    }
                )
                sum_duration += duration
                if sum_limit_duration == -1:
                    continue
                if sum_duration >= sum_limit_duration:
                    break

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, id):
        key, audio, transcript = (
            self.manifest[id]["key"],
            self.manifest[id]["audio"],
            self.manifest[id]["transcript"],
        )

        # wav -> mel
        mel = pad_or_trim(log_mel_spectrogram(audio), N_FRAMES)
        text = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(transcript)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            "utt_id": key,
        }


class WhisperFolderNoTextASRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir_root: str,
        tokenizer: Tokenizer,
        max_duration: float = 30,
        min_duration: float = 0.1,
        sum_limit_duration: float = -1,
    ):

        super().__init__()
        self.tokenizer = tokenizer
        self.manifest = []
        self.dir_root = dir_root
        sum_duration = 0
        for f_audio in Path(dir_root).glob("*.wav"):
            f_wav_name_only = f_audio.stem
            duration = 30

            transcript = "A"
            self.manifest.append(
                {
                    "key": f_wav_name_only,
                    "audio": os.path.join(dir_root, f_audio),
                    "duration": duration,
                    "transcript": transcript,
                }
            )
            sum_duration += duration
            # if sum_limit_duration == -1:
            #     continue
            # if sum_duration >= sum_limit_duration:
            #     break

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, id):
        key, audio, transcript = (
            self.manifest[id]["key"],
            self.manifest[id]["audio"],
            self.manifest[id]["transcript"],
        )

        # wav -> mel
        mel = pad_or_trim(log_mel_spectrogram(audio), N_FRAMES)
        text = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(transcript)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            "utt_id": key,
        }


class WhisperASRDataCollator:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, utt_ids = [], [], [], []
        for feature in features:
            input_ids.append(feature["input_ids"])
            labels.append(feature["labels"])
            dec_input_ids.append(feature["dec_input_ids"])
            utt_ids.append(feature["utt_id"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(label) for label in labels]
        dec_input_ids_length = [len(dec_input_id) for dec_input_id in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), "constant", constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        batch = {"labels": labels, "dec_input_ids": dec_input_ids}

        batch = {
            k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()
        }
        batch["input_ids"] = input_ids
        batch["utt_id"] = utt_ids

        return batch


def valid_audio_text_safe(text, audio, text_max_length, audio_max_sample_length):
    if len(text) == 0:
        return False
    if len(text) > text_max_length:
        return False
    if audio is None:
        return False
    if len(audio) > audio_max_sample_length:
        return False
    return True


def save_data_list(data_list: list, list_path: Union[Path, str]):
    with open(list_path, "w") as f:
        f.writelines("\t".join(x) + "\n" for x in data_list)


def load_data_list(list_path: Union[Path, str]):
    return [x.strip("\n").split("\t") for x in open(list_path, "r").readlines()]


def load_audio(file: str, start_sec: float, end_sec: float, sr: int = 16000):

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0, ss=start_sec, to=end_sec)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
