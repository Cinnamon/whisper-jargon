import csv
import re

import jaconv
import jiwer

dictionary = []
result_result = {}
groundtruth = {}

hiragana_full = r"[ぁ-ゟ]"
katakana_full = r"[゠-ヿ]"
kanji = r"[㐀-䶵一-鿋豈-頻]"
radicals = r"[⺀-⿕]"
katakana_half_width = r"[｟-ﾟ]"
alphanum_full = r"[！-～]"
symbols_punct = r"[、-〿]"
misc_symbols = r"[ㇰ-ㇿ㈠-㉃㊀-㋾㌀-㍿]"
ascii_char = r"[ -~]"


def extract_unicode_block(unicode_block, string):
    """extracts and returns all texts from a unicode block from string argument.
    Note that you must use the unicode blocks defined above, or patterns of similar form"""
    return re.findall(unicode_block, string)


def remove_unicode_block(unicode_block, string):
    """removes all chaacters from a unicode block and returns all remaining texts from string argument.
    Note that you must use the unicode blocks defined above, or patterns of similar form"""
    return re.sub(unicode_block, "", string)


if __name__ == "__main__":
    with open(
        "D:\\srqu_report-materials\\ref1.csv", encoding="utf8", newline=""
    ) as f_csv:
        reader = csv.reader(f_csv, delimiter="\t")
        for row in reader:
            groundtruth[row[0]] = row[1]

    with open(
        "D:\\srqu_report-materials\\hypo_no_dict_small.csv", encoding="utf8", newline=""
    ) as f_csv:
        reader = csv.reader(f_csv, delimiter="\t")
        for row in reader:
            result_result[row[0]] = row[1]

    total_no_dict = 0
    total_with_dict = 0
    total = 0
    freq_dict = {}

    for file_name, gt in groundtruth.items():
        current_gt = gt.replace("。", "").replace("、", "")
        current_gt = jaconv.normalize(current_gt)
        hypothesis = result_result[file_name].replace("。", "").replace("、", "")
        hypothesis = jaconv.normalize(hypothesis)
        if len(current_gt) == 0:
            continue
        if len(hypothesis) == 0:
            continue
        gt_vs_no_dict_output = jiwer.process_characters(current_gt, hypothesis)
        for alignment in gt_vs_no_dict_output.alignments[0]:
            if alignment.type == "substitute":

                word = current_gt[alignment.ref_start_idx : alignment.ref_end_idx]
                wrong_word = hypothesis[alignment.hyp_start_idx : alignment.hyp_end_idx]
                # print(alignment, word, wrong_word)
                # if word == remove_unicode_block(kanji, word):
                #     continue
                if len(word) == 1:
                    continue
                if word[0].isdigit():
                    continue
                if "" == remove_unicode_block(hiragana_full, word):
                    continue
                if word not in freq_dict:
                    freq_dict[word] = 1
                else:
                    freq_dict[word] += 1
            elif alignment.type == "delete":
                word = current_gt[alignment.ref_start_idx : alignment.ref_end_idx]
                # if word == remove_unicode_block(kanji, word):
                #     continue
                if len(word) == 1:
                    continue
                if word[0].isdigit():
                    continue
                if "" == remove_unicode_block(hiragana_full, word):
                    continue
                if word not in freq_dict:
                    freq_dict[word] = 1
                else:
                    freq_dict[word] += 1

    print(freq_dict)
    for k1 in freq_dict.keys():
        for k2 in freq_dict.keys():
            if k2 in k1 and k2 != k1:
                freq_dict[k2] = 0
    final_dict = dict(sorted(freq_dict.items()))
    filtered_ban_dict = [k for k, v in final_dict.items() if v > 1]
    with open(
        "recipes/srqu/srqu_boost_dictionary_algor_small.csv",
        encoding="utf8",
        mode="w",
        newline="",
    ) as f_csv:
        for x in filtered_ban_dict:
            f_csv.write(f"{x}\n")
    # cer_metrics = evaluate.load("cer")
    # cer_no_dict = cer_metrics.compute(references=ref_no_dict, predictions=hyp_no_dict)
    # cer_with_dict = cer_metrics.compute(references=ref_with_dict, predictions=hyp_with_dict)
    # print(f"CER: {cer_no_dict}, CER_dict: {cer_with_dict} ")
