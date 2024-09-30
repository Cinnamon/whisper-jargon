import csv
import string
import sys
from pathlib import Path

import evaluate
import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
from whisper_finetune.dataset import WhisperASRDataCollator, WhisperASRDataset
from whisper_finetune.model import WhisperModelModule

from whisper_main import whisper
from whisper_main.whisper.transcribe import transcribe


def inference():
    # load config
    config_path = Path("config.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # dirs and paths
    Path(config["path"]["checkpoint"])
    with_timestamps = bool(config["data"]["timestamps"])
    prompt_str = ""
    if config["data"]["dict_path"] is not None and config["inference"]["using_prompt"]:
        with open(
            config["data"]["dict_path"], "r", encoding="utf8", newline=""
        ) as f_dict:
            reader = csv.reader(f_dict)
            for row in reader:
                prompt_str += ", "
                dict_word = row[0]
                prompt_str += f"{dict_word}"
            prompt_str += (
                "Generate a text that contains all the words in the dictionary."
            )
    else:
        prompt_str = None

    # device = "gpu" if torch.cuda.is_available() else "cpu"
    # prompt_str = 'はい、日本語で漏えい、当該、塔、漏洩、ガス、配管、熱交換器、脱硫、箇所、器、気密、水添、軽油、縁切り、液、発、MPA、反応器、材、冷媒、覚知、継手、蒸発器、臭気、後、漏れ、ブタジエン、要領、内、槽、冷凍機、定、修、定常、貯槽、閉止、量、質、臭、事、側、充てん、出火、増し締め、部、架台、法、減、報、計、計器、Kg、石、雨水、改、イソブチレン、払出、押え、フロン、ローリー、重合、時、下部、機、五十、個所、吐出、入、冷却器、分、厚、取付け、同日、基、安全弁、温、管、浸入、為、置換、非、遮断、ｍｍ、けん水、その後、錆、高圧ガス、泡、溶射、ナフサ、パッ、キン、締結、フランジ、フレア、ブタジエンガス、プラント、及び、移、試験、圧縮機、仕込、係員、保温、入口、系、内層、原因、原料、取付、受入、化、頂、増し、締め、運転、導管、小火、工事、廃、応力、応力腐食割れ、接触改質、業者、機器、気体、水素化脱硫装置、点検、無い、ｓｕｓ、災、直ちに、孔、破断、窒素、経年、締付け、膨張、計装、誤、課員、近傍、逆止弁、進行、重合反応、銅管、銘柄、除、害、リング の単語をすべて含むテキストを生成します。'

    # tools
    print(f"Using prompt: {prompt_str}")
    print(f"Using dict path: {config['data']['dict_path']}")
    whisper_options = whisper.DecodingOptions(
        language=config["data"]["lang"],
        without_timestamps=not with_timestamps,
        beam_size=5,
        prompt=prompt_str,
        dict_path=config["data"]["dict_path"],
        dict_coeff=config["inference"]["dict_coeff"],
        ngram_path=None,
        ngram_coeff=0.001,
    )
    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True, language=config["data"]["lang"], task=whisper_options.task
    )

    # list
    # dataset = WhisperASRDataset(config["test_manifest"], config["test_root"], whisper_tokenizer)
    # dataset= WhisperFolderNoTextASRDataset(config["test_root"], whisper_tokenizer)
    dataset = WhisperASRDataset(
        config["test_manifest"], config["test_root"], whisper_tokenizer
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=WhisperASRDataCollator()
    )

    # load models
    config["inference"]["epoch_index"]
    # checkpoint_path = checkpoint_dir / "checkpoint" / f"srqu-checkpoint-epoch={epoch:04d}.ckpt"
    # state_dict = torch.load(checkpoint_path)
    # state_dict = state_dict['state_dict']
    whisper_model = WhisperModelModule(
        config["train_manifest"],
        config["train_root"],
        config["val_manifest"],
        config["val_root"],
        config["train"],
        model_name=config["model_name"],
        lang=config["data"]["lang"],
    )
    # whisper_model.load_state_dict(state_dict)

    # inference
    ref, hyp, utt_ids = [], [], []

    # for b in tqdm(loader):
    #     input_id = b["input_ids"].half().cuda()
    #     label = b["labels"].long().cuda()
    #     utt_id = b["utt_id"]
    #     with torch.no_grad():
    #
    #         # # hypothesis = whisper_model.model.decode(input_id, whisper_options)
    #         #
    #         # for h in hypothesis:
    #         #     hyp.append(h.text.lower())
    #         # print(input_id.size())
    #         for i in range(input_id.shape[0]):
    #             hypothesis = transcribe(whisper_model.model, input_id[i],
    #                                     initial_prompt=prompt_str,
    #
    #                                     language=config["data"]["lang"],
    #                                     dict_path=config["data"]["dict_path"],
    #                                     dict_coeff=config["inference"]["dict_coeff"],
    #                                     beam_size=5,
    #                                     task="transcribe",
    #                                     fp16=True
    #                                     )
    #             hyp.append(hypothesis["text"].lower())
    #
    #         for l in label:
    #             l[l == -100] = whisper_tokenizer.eot
    #             r = whisper_tokenizer.decode(l)
    #             for special_token in whisper_tokenizer.special_tokens:
    #                 r = r.replace(special_token, "")
    #             ref.append(r.lower())
    #         for id in utt_id:
    #             utt_ids.append(id)

    # ======================================================
    for i in tqdm(range(len(dataset))):
        audio_path = dataset.manifest[i]["audio"]
        transcript = dataset.manifest[i]["transcript"]
        hypothesis = transcribe(
            whisper_model.model,
            audio_path,
            initial_prompt=prompt_str,
            language=config["data"]["lang"],
            dict_path=config["data"]["dict_path"],
            dict_coeff=config["inference"]["dict_coeff"],
            beam_size=5,
            task="transcribe",
            fp16=True,
            ngram_path=None,
            ngram_coeff=0.001,
        )
        hyp.append(hypothesis["text"].lower())
        ref.append(transcript)

        utt_ids.append(dataset.manifest[i]["key"])
    # ======================================================

    # device = whisper_model.model.device
    # for b in tqdm(loader):
    #     input_id = b["input_ids"].half().to(device)
    #     label = b["labels"].long().to(device)
    #     utt_id = b["utt_id"]
    #     with torch.no_grad():
    #         hypothesis = whisper_model.model.decode(input_id, whisper_options)
    #         for h in hypothesis:
    #             hyp.append(h.text)

    #         for l in label:
    #             l[l == -100] = whisper_tokenizer.eot
    #             r = whisper_tokenizer.decode(l)
    #             for special_token in whisper_tokenizer.special_tokens:
    #                 r = r.replace(special_token, "")
    #             ref.append(r)
    #         for id in utt_id:
    #             utt_ids.append(id)

    # ======================================================

    with open("hypo_ngram_small_no_dict_2.csv", "w+") as fo:
        for id, h in zip(utt_ids, hyp):
            fo.write(id + "\t" + h + "\n")
    with open("ref_ngram_small_no_dict_2.csv", "w+") as fo:
        for id, r in zip(utt_ids, ref):
            fo.write(id + "\t" + r + "\n")

    # compute CER
    wer_metrics = evaluate.load("wer")
    wer = wer_metrics.compute(references=ref, predictions=hyp)
    print(f"WER: {wer}")


def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


if __name__ == "__main__":
    inference()
    # wer_metrics = evaluate.load("wer")
    # normalizer = EnglishTextNormalizer()

    # hyps_data = read_two_column_data("hypo_ngram_small_no_dict_2.csv")
    # refs_data = read_two_column_data("ref_ngram_small_no_dict_2.csv")

    # hyps, refs = [], []
    # for hyp, ref in zip(hyps_data, refs_data):
    #     normalized_hyp = normalizer(hyp[1])
    #     normalized_ref = normalizer(ref[1])
    #     hyps.append(normalized_hyp)
    #     refs.append(normalized_ref)

    # wer = wer_metrics.compute(references=refs, predictions=hyps)
    # print(f"WER: {wer}")
