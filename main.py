import glob
import os
import re
from copy import copy
from difflib import ndiff

from matplotlib.pyplot import pause

from alignment_algo import make_most_probable_syntagma_distribution, \
    make_most_probable_syntagma_distribution_2
from sig_analysis import detect_pauses, detect_allophone_types
from utils.json_utils import get_object_from_json
from utils.signal_classes import Signal, Seg, Label


def process_signal(signal: Signal, config: dict) -> list:
    """
    detects cluster boundaries in audio
    writes seg-file with word boundaries labels
    """
    labels = [Label(0, "Y1", 'begin')]
    labels = detect_pauses(signal, labels, config)
    labels = detect_allophone_types(signal, labels, config)

    return labels


char_dict = {
    'а': ['a'],
    'б': ['b', 'p', "b'"],
    'в': ['v', 'f', "v'"],
    'г': ['g', 'k', "g'"],
    'д': ['d', 't', "d'"],
    'е': ['e', 'j e'],
    'ё': ['o', 'j o'],
    'ж': ['zh', '', "zh'"],
    'з': ['z', 's', "z'"],
    'и': ['i'],
    'й': ['j', '', 'j'],
    'к': ['k', '', "k'"],
    'л': ['l', '', "l'"],
    'м': ['m', '', "m'"],
    'н': ['n', '', "n'"],
    'о': ['o'],
    'п': ['p', '', "p'"],
    'р': ['r', '', "r'"],
    'с': ['s', '', "s'"],
    'т': ['t', 't', "t'"],
    'у': ['u'],
    'ф': ['f', '', "f'"],
    'х': ['h', '', "h'"],
    'ц': ['c', '', 'c'],
    'ч': ['ch', '', 'ch'],
    'ш': ['sh', '', 'sh'],
    'щ': ['sc', '', 'sc'],
    'ъ': [''],
    'ы': ['y'],
    'ь': [''],
    'э': ['e'],
    'ю': ['u', 'j u'],
    'я': ['a', 'j a'],
}

stop_signs = ('.', ',', ':', ';', '!', '?')
voiced_cons = ('б', 'в', 'г', 'д', 'ж', 'з', 'й', 'л', 'м', 'н', 'р')
unvoiced_cons = ('к', 'п', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ')
vowels = ('а', 'э', 'о', 'у', 'ы', 'и', 'е', 'ё', 'ю', 'я')
softening_vowels = ('и', 'е', 'ё', 'ю', 'я')


def translate_to_latin(text: str) -> str:
    latin_text = "0"

    text_ = text.lower().strip().replace(' ', '0')

    for i, ch in enumerate(text_):
        if ch in vowels:
            if ch in tuple("еёюя") and (i == 0 or text_[i - 1] in vowels or text_[i - 1] == '0' or text_[i - 1] == 'ь'):
                latin_text += char_dict[ch][1] + " "
            else:
                latin_text += char_dict[ch][0] + " "

        elif (ch in voiced_cons or ch in unvoiced_cons) and i < len(text_) - 1 and (
                text_[i + 1] in softening_vowels or text_[i + 1] == 'ь'):
            latin_text += char_dict[ch][2] + " "
            continue
        # TODO Озвончение глухих согласных перед звонкими, в том числе на границе слов.
        elif ch in ('б', 'в', 'г', 'д', 'з'):
            if i < len(text_) - 1:
                if text_[i + 1] in stop_signs or text_[i + 1] in unvoiced_cons or (
                        text_[i + 1] == '0' and text_[i + 2] in unvoiced_cons):
                    latin_text += char_dict[ch][1] + " "
                else:
                    latin_text += char_dict[ch][0] + " "
            else:
                latin_text += char_dict[ch][0] + " "
        elif ch not in char_dict.keys():
            latin_text += ch + " "
        else:
            latin_text += char_dict[ch][0] + " "
    return latin_text[1:]


def process_text(text: str, for_syntagmas) -> (list, list):
    """
    detects cluster boundaries in text
    0 - vowels and sonorants
    1 - noisy
    2 - voiceless stops
    3 - other consonants
    """
    latin_txt = translate_to_latin(text)  # точка для добавления паузы в начале

    vowels_and_sonorant = ('a', 'e', 'i', 'u', 'o', 'y', 'l', 'm', 'n', 'v', "l'", "m'", "n'", "r'", "v'", "r", "j")
    stops = ('p', 't', 'k')
    affricates = ("p'", "k'", "t'", "d'", "c")
    noisy = ('z', "z'", 'zh', 's', 'f', 'h', "s'", "f'", "h'", 'ch', 'sh', 'sc', "CH", "ch_", "zh'")
    other = ('b', 'd', "b'", 'g', "g'")

    clusters = [(0, 9)]
    word_boundaries_indexes = [0]

    txt_arr = latin_txt.lower().strip().split()
    for i, ch in enumerate(txt_arr):
        if ch == '0':
            word_boundaries_indexes.append(i)  # -1 из-за границы слова в начале
        elif ch in vowels_and_sonorant and clusters[-1][1] != 0:
            clusters.append((i, 0))
        elif ch in noisy and (clusters[-1][1] != 1 or (i != 0 and txt_arr[i - 1] == "0")):
            clusters.append((i, 1))
        elif ch in stops and (clusters[-1][1] != 2 or (i != 0 and txt_arr[i - 1] == "0")):
            clusters.append((i, 2))
        elif ch in affricates and (clusters[-1][1] != 2 or (i != 0 and txt_arr[i - 1] == "0")):
            if txt_arr[i - 1] not in affricates:
                clusters.append((i, 2))
                clusters.append((i, 1))
        elif ch in other and (clusters[-1][1] != 3 or (i != 0 and txt_arr[i - 1] == "0")):
            if for_syntagmas:
                clusters.append((i, 0))
            else:
                clusters.append((i, 3))

    clusters = clusters[1:]
    return clusters, word_boundaries_indexes, txt_arr


lbl_to_str_dict = {
    "": "4",
    "noisy": "1",
    "periodic": "0",
    "other": "3",
    "voiceless_stop": "2",
    "begin": "",
    "new_synt": "",
    "pause": "4"
}


def convert_ac_labels_to_string(ac_labels):
    ac_string = ""
    for label in ac_labels:
        if label.level == "R1":
            ac_string += lbl_to_str_dict[label.text]
    return ac_string


def unite_txt_clusters(txt_clusters_synt_new, unite_by_position=False):
    txt_clusters_synt_new_un = []
    delete_next = False
    for cl1, cl2 in zip(txt_clusters_synt_new, txt_clusters_synt_new[1:]):
        if delete_next:
            delete_next = False
            continue
        if cl2[1] == cl1[1] or (unite_by_position and cl2[0] == cl1[0] and  cl2[1] != cl1[1]): # для двухчастных согласных
            txt_clusters_synt_new_un.append(cl1)
            delete_next = True
        else:
            txt_clusters_synt_new_un.append(cl1)
            delete_next = False
    txt_clusters_synt_new_un.append(txt_clusters_synt_new[-1])
    return txt_clusters_synt_new_un


def rewrite_acoustic_markers(best_syntagma_distribution, best_operations, ac_labels, txt_arr, txt_clusters):
    new_txt_distribution = []

    for synt_count in range(len(best_operations)):
        synt_distr = best_syntagma_distribution[synt_count]
        operations = best_operations[synt_count]
        txt_clusters_synt = list(filter(lambda x: synt_distr[0] <= x[0] < synt_distr[1], txt_clusters))

        if len(operations) == 0:
            new_txt_distribution.extend(txt_clusters_synt)
        else:
            insertions_count = 0
            for (operation, index, ch) in operations:
                txt_clusters_synt_new = copy(txt_clusters_synt)
                index = index + insertions_count
                if operation == 'd':
                    to_del_el = txt_clusters_synt_new[index]
                    txt_clusters_synt_new = txt_clusters_synt_new[:index] + txt_clusters_synt_new[index + 1:]
                    txt_clusters_synt_new.insert(index, (to_del_el[0], txt_clusters_synt_new[index][1]))
                    insertions_count += 1
                    txt_clusters_synt_new = unite_txt_clusters(txt_clusters_synt_new)
                elif operation == 'r':
                    txt_clusters_synt_new[index] = (txt_clusters_synt_new[index][0], int(ch))
                    txt_clusters_synt_new = unite_txt_clusters(txt_clusters_synt_new)
                elif operation == 'i':
                    txt_clusters_synt_new = txt_clusters_synt_new[:index] + [
                        (txt_clusters_synt_new[index][0], int(ch))] + txt_clusters_synt_new[index:]
                    txt_clusters_synt_new = unite_txt_clusters(txt_clusters_synt_new)

            txt_clusters_synt_new = unite_txt_clusters(txt_clusters_synt_new, True)

            new_txt_distribution.extend(txt_clusters_synt_new)

    pause_starts = [x[1] for x in best_syntagma_distribution]

    # spaces_to_insert = []
    # for cl1, cl2 in zip(txt_clusters, txt_clusters[1:]):
    #     for pause_start in pause_starts:
    #         if cl1[0] < pause_start < cl2[0]:
    #             spaces_to_insert.append((txt_clusters.index(cl2) + len(spaces_to_insert), (pause_start, cl1[1])))
    #
    # for (index, new_label) in spaces_to_insert:
    #     new_txt_distribution.insert(index, new_label)

    ac_labels_count = 0
    prev_label = ""
    new_ac_labels = []

    for i, label in enumerate(ac_labels):
        if label.level == 'R1' and label.text != "":
            label_text = " ".join(
                txt_arr[new_txt_distribution[ac_labels_count][0]: new_txt_distribution[ac_labels_count + 1][0]])

            label_text = re.sub(r'[\d.,!:;-]', '', label_text)
            new_ac_labels.append(Label(label.position, "R2", label_text))
            if label_text in ('p', 't', 'k') and prev_label == "" and label.text != "voiceless_stop":
                ac_labels_count += 1
                next_label_text = " ".join(
                txt_arr[new_txt_distribution[ac_labels_count][0]: new_txt_distribution[ac_labels_count + 1][0]])
                next_label_text = re.sub(r'[\d.,!:;-]', '', next_label_text)
                new_ac_labels.append(Label(label.position + 512, "R2", next_label_text))
            ac_labels_count += 1
            prev_label = label.text

    move_index = 0
    for label1, label2 in zip(new_ac_labels, new_ac_labels[1:]):
        affricates = ("p'", "k'", "t'", "d'", "c")
        if label1.text.strip() in affricates and label2.text.strip() in affricates:
            label1.text = label1.text.strip() + " " + label2.text.strip()
            label1.position = new_ac_labels[new_ac_labels.index(label1) + move_index].position
            move_index += 1
        else:
            label1.text = new_ac_labels[new_ac_labels.index(label1) + move_index].text

    ac_labels.extend(new_ac_labels)
    return ac_labels


def define_syntagmas_2(ac_labels, txt_clusters, text, word_boundaries_indexes, sampling_freq, txt_arr):
    ac_part_labels = []
    synt_labels = []
    ac_synt_durations = []

    for label in ac_labels:
        if label.level == "R1":
            ac_part_labels.append(label)
        elif label.level == "Y1":
            synt_labels.append(label)

    ac_string = convert_ac_labels_to_string(ac_part_labels)

    for label1, label2 in zip(synt_labels, synt_labels[1:]):
        if label1.text == "new_synt":
            ac_synt_durations.append((label2.position - label1.position) / sampling_freq)

    best_syntagma_distribution, best_operations_all = make_most_probable_syntagma_distribution_2(ac_string,
                                                                                                 txt_clusters,
                                                                                                 word_boundaries_indexes,
                                                                                                 ac_synt_durations,
                                                                                                 txt_arr)
    count = 0
    for label in ac_labels:
        if label.level == "Y1" and label.text == "new_synt":
            synt_text = text[best_syntagma_distribution[count][0]: best_syntagma_distribution[count][-1]]
            label.text = synt_text
            count += 1

    rewrite_acoustic_markers(best_syntagma_distribution, best_operations_all, ac_labels, txt_arr, txt_clusters)

    return ac_labels


def main(wav_fn, text_fn) -> None:
    """
    writes seg-file with word boundaries
    :param wav_fn: audio filename
    :param text_fn: correspondent text
    """
    text = ""
    with open(text_fn, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        text += line.strip()

    signal = Signal(wav_fn)
    signal.read_sound_file()

    config = get_object_from_json("config.json")

    # 1. Получение акустических меток
    ac_labels = process_signal(signal, config)
    ac_labels = sorted(ac_labels, key=lambda x: x.position)
    print("Получены акустические метки")

    # 2. Получение псевдотранскрипции и границ слов
    txt_clusters, word_boundaries_indexes, txt_arr = process_text(text, True)
    print("Получена псевдотранскрипция", txt_clusters)

    # 3. Сопоставление акустических меток с текстом, поиск пауз
    # ac_labels = define_syntagmas(ac_labels, txt_clusters, text)
    ac_labels = define_syntagmas_2(ac_labels, txt_clusters, text, word_boundaries_indexes, signal.params.samplerate,
                                   txt_arr)
    print(f"Получены метки с разбиением на синтагмы")

    # 4 Более точный ак анализ и транскрипция внутри каждой синтагмы

    # 5 Выравнивание и определение границ слов

    # # 4. Выравнивание внутри синтагмы, вычисление кластеров с количеством пробелов на них
    # ac_parts_with_space = make_alignment(acoustic_parts, text_parts, word_boundaries_indexes)
    #
    # # 5. Нахождение границ слов с дополнительным акустическим анализом
    # words = re.sub(r'[.,?!;:]', ' пауза', text).split(" ")
    # ac_labels = define_word_boundaries(ac_parts_with_space, ac_labels, words)

    # 6. Запись сег файла
    new_seg_fn = wav_fn.split(".")[0] + ".seg"
    # new_seg = Seg(new_seg_fn, word_boundaries, signal.params)
    new_seg = Seg(new_seg_fn, ac_labels, signal.params)
    new_seg.write_seg_file()
    print(f"Границы слов записаны в файл {new_seg_fn}")


wav_fn = r"D:\pycharm_projects\word_segmentator\data\source_data\av15t.wav"
text_fn = r"D:\pycharm_projects\word_segmentator\data\source_data\av15t.txt"

main(wav_fn, text_fn)

# fld_name = r"D:\test_andre"
# wav_files = glob.glob(f"{fld_name}/*.wav", recursive=True)
# for file in wav_files:
#     try:
#         text_fn = os.path.splitext(file)[0] + ".txt"
#         main(file, text_fn)
#     except Exception:
#         print(f"{file} got error")
