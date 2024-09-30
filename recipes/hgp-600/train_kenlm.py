import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))

from whisper_main import whisper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--kenlm_bin_path",
    help="",
    type=str,
    default="/home/barb_cinnamon_is/kenlm/build/bin",
)
parser.add_argument(
    "--input_path",
    help="",
    type=str,
    default="/home/barb_cinnamon_is/HCP-600-IPhone6L-parsed/output.txt",
)
parser.add_argument(
    "--whispered_path",
    help="",
    type=str,
    default="/home/barb_cinnamon_is/HCP-600-IPhone6L-parsed/output_whispered.txt",
)
parser.add_argument(
    "--arpa_output_path",
    help="",
    type=str,
    default="/home/barb_cinnamon_is/HCP-600-IPhone6L-parsed/kenlm.arpa",
)
parser.add_argument(
    "--model_output_path",
    help="",
    type=str,
    default="/home/barb_cinnamon_is/HCP-600-IPhone6L-parsed/hgp-600.bin",
)


def train_kenlm(args):

    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True, language="ja", task="transcribe"
    )
    with open(args.input_path, "r", encoding="utf8") as fi:
        with open(args.whispered_path, "w", encoding="utf8") as fo:
            for line in fi:
                line = line.replace("\n", "").replace("\r", "").replace(" ", "")
                line_modified = whisper_tokenizer.encode(line)
                # print(line_modified, line)
                fo.write(" ".join([str(x) for x in line_modified]) + "\n")

    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "lmplz"),
        "-o",
        "5",
        "--arpa",
        args.arpa_output_path,
        "--discount_fallback",
    ]
    first_process_args = ["cat"] + [args.whispered_path]
    first_process = subprocess.Popen(
        first_process_args, stdout=subprocess.PIPE, stderr=sys.stderr
    )

    kenlm_p = subprocess.run(
        kenlm_args,
        stdin=first_process.stdout,
        capture_output=False,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    first_process.wait()

    if kenlm_p.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")

    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        args.arpa_output_path,
        args.model_output_path,
    ]

    ret = subprocess.run(
        kenlm_args,
        capture_output=False,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")


if __name__ == "__main__":
    args = parser.parse_args()
    train_kenlm(args)
