import re

from hy_utils import read_two_column_data, JapaneseTextNormalizer
from whisper_main.whisper.normalizers import EnglishTextNormalizer


def check_valid_jp_word(word):
    return re.search("[一-龯ぁ-ゖァ-ヺ０-９Ａ-Ｚａ-ｚ々ーa-zA-Z0-9]", word)

def calculate_drr(dictionary, references, predictions):
    # Initialize counters
    total_should_recognize = 0
    total_correctly_recognized = 0

    # Iterate over each reference and prediction
    for ref, pred in zip(references, predictions):
        # Split the reference and prediction into words
        ref_words = ref.split(" ")
        pred_words = pred.split(" ")

        # Iterate over each word in the dictionary
        for word in dictionary:
            # If the word should be recognized (it's in the reference), increment the counter
            word = word.lower()
            if word in ref_words:
                count_in_ref = ref.count(word)
                total_should_recognize += count_in_ref

            # If the word is correctly recognized (it's also in the prediction), increment the counter
            if word in pred_words:
                count_in_pred = pred.count(word)
                total_correctly_recognized += count_in_pred

    # Calculate and return the DRR
    if total_should_recognize == 0:
        return None  # Avoid division by zero
    else:
        return total_correctly_recognized / total_should_recognize

if __name__=="__main__":
    # jp_normalizer = JapaneseTextNormalizer()
    en_normalizer = EnglishTextNormalizer()

    # Libri
    dict_path = "recipes/libri/libri_boost_dictionary_algor.csv"

    # medium
    pred_path = "recipes/libri/hypo_ngram_small_no_dict_2.csv"
    gold_path = "recipes/libri/ref_ngram_small_no_dict_2.csv"


    gold_data = read_two_column_data(gold_path)
    pred_data = read_two_column_data(pred_path)
    assert len(gold_data) == len(pred_data), f"Length mismatch: {len(gold_data)} vs {len(pred_data)}"

    lines = open(dict_path, mode="r", encoding="utf-8").readlines()
    # word_dict = [line.strip() for line in lines if check_valid_jp_word(line.strip())]
    word_dict = [line.strip().lower() for line in lines]
    print(f"Found {len(word_dict)} words in {dict_path}")

    predictions = []
    references = []

    for gold_utt, pred_utt in zip(gold_data, pred_data):
        assert gold_utt[0] == pred_utt[0], f"Utterance names do not match: {gold_utt[0]} vs {pred_utt[0]}"
        references.append(en_normalizer(gold_utt[1]))
        predictions.append(en_normalizer(pred_utt[1]))


    drr = calculate_drr(word_dict, references, predictions)
    print(f"DRR: {drr}")
