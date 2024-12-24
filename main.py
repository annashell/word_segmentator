import glob
import os
import re
from copy import copy
from difflib import ndiff

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
    # TODO звонкие мягкие смычные = период + шум
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


def nearest(lst, target):
    return min(lst, key=lambda x: abs(x - target))


def convert_ac_labels_to_string(ac_labels):
    ac_string = ""
    for label in ac_labels:
        if label.level == "R1":
            ac_string += lbl_to_str_dict[label.text]
    return ac_string


def convert_txt_clusters_to_string(text_clusters):
    txt_str = "4"
    for cluster in text_clusters:
        txt_str += str(cluster[1])
    return txt_str


def make_alignment(ac_parts, text_parts, word_boundaries_indexes) -> list:
    """
    makes text and audio labels alignment
    :return list of dicts, where key is a word, value - start and end of the word in audio
    """
    ac_parts_with_space = {}

    for i, part in enumerate(ac_parts):
        ac_string = convert_ac_labels_to_string(part)
        text_part = text_parts[i]
        txt_string = convert_txt_clusters_to_string(text_part)
        str_difference = list(ndiff(ac_string, txt_string))

        # индексы текстовых кластеров с пробелами
        spaced_parts_indexes = []
        for ind in word_boundaries_indexes:
            for cl1, cl2 in zip(text_part, text_part[1:]):
                if ind == cl2[0] - 1:
                    spaced_parts_indexes.append(text_part.index(cl2))
                elif cl1[0] <= ind < cl2[0]:
                    spaced_parts_indexes.append(text_part.index(cl1))

        ac_part_labels = []
        for label in part:
            if label.level == "R1":
                ac_part_labels.append(label)

        cnt = -1
        plus_count = 0
        for j, x in enumerate(str_difference):
            if x.startswith('+'):
                plus_count += 1
            if not x.startswith('-'):
                cnt += 1
                if cnt in spaced_parts_indexes:
                    number_of_spaces = spaced_parts_indexes.count(cnt)
                    ac_spaced_part_index = j - plus_count + 1
                    ac_part_with_space = ac_part_labels[ac_spaced_part_index]
                    ac_part_with_space_st = ac_part_with_space.position
                    if ac_part_with_space_st not in ac_parts_with_space.keys():
                        ac_parts_with_space[ac_part_with_space_st] = number_of_spaces
                    else:
                        ac_parts_with_space[ac_part_with_space_st] += number_of_spaces

        ac_parts_with_space[part[-1].position] = 1
        if i < len(ac_parts) - 1:
            ac_parts_with_space[ac_parts[i + 1][0].position] = 1

    return ac_parts_with_space


def split_acoustic_labels(ac_labels):
    acoustic_parts = []
    new_part = []
    for label in ac_labels:
        if label.text == "pause":
            acoustic_parts.append(new_part)
            new_part = []
        else:
            new_part.append(label)
    acoustic_parts = list(filter(lambda x: len(x) > 0, acoustic_parts))
    return acoustic_parts


def split_text_clusters(txt_clusters):
    text_parts = []
    new_text_part = []
    for cluster in txt_clusters:
        if cluster[1] == 4:
            text_parts.append(new_text_part)
            new_text_part = []
        else:
            new_text_part.append(cluster)
    text_parts = list(filter(lambda x: len(x) > 0, text_parts))
    return text_parts


def define_word_boundaries(ac_parts_with_space, ac_labels, words):
    space_labels = []
    cnt = 0

    filtered_ac_labels = list(filter(lambda x: x.position in ac_parts_with_space.keys(), ac_labels))

    for label, next_label in zip(filtered_ac_labels, filtered_ac_labels[1:]):
        if label.position in ac_parts_with_space.keys():
            if cnt >= len(words):
                break
            number_of_spaces = ac_parts_with_space[label.position]
            cluster_duration = next_label.position - label.position
            for i in range(number_of_spaces):
                space_labels.append(
                    Label(label.position + (i + 1) * (cluster_duration // (number_of_spaces + 1)), "B1", words[cnt]))
                cnt += 1

    ac_labels.extend(space_labels)
    return ac_labels


def define_syntagmas(ac_labels, txt_clusters, text):
    ac_part_labels = []
    for label in ac_labels:
        if label.level == "R1":
            ac_part_labels.append(label)

    ac_string = convert_ac_labels_to_string(ac_part_labels)
    text_string = convert_txt_clusters_to_string(txt_clusters)
    str_difference = list(ndiff(text_string, ac_string))

    alignments = pairwise2.align.globalxx(text_string.split(),
                                          ac_string.split(),
                                          gap_char=['-'])

    format_alignment(*alignments[0])

    # индексы текстовых кластеров, после которых идет пауза
    spaced_parts_indexes = []
    without_sign_count = 0
    for dif in str_difference:
        if not dif.startswith(("+", "-")):
            without_sign_count += 1
        if dif == "+ 4":
            spaced_parts_indexes.append(without_sign_count)

    pause_indexes = [txt_clusters[x][0] for x in spaced_parts_indexes]

    word_boundaries_indexes = []
    for i, ch in enumerate(list(text)):
        if ch == " ":
            word_boundaries_indexes.append(i)

    pause_indexes_final = [0]
    for ind in pause_indexes:
        pause_indexes_final.append(nearest(word_boundaries_indexes, ind))
    pause_indexes_final.append(len(text))

    text_fragments = []
    for ind1, ind2 in zip(pause_indexes_final, pause_indexes_final[1:]):
        text_frag = text[ind1: ind2]
        text_fragments.append(text_frag)

    lbl_count = 0
    for label in ac_labels:
        if label.level == "Y1" and label.text in ("begin", "new_synt"):
            label.text = text_fragments[lbl_count]
            lbl_count += 1

    return ac_labels


def unite_txt_clusters(txt_clusters_synt_new):
    txt_clusters_synt_new_un = []
    delete_next = False
    for cl1, cl2 in zip(txt_clusters_synt_new, txt_clusters_synt_new[1:]):
        if delete_next:
            delete_next = False
            continue
        if cl2[1] == cl1[1]:
            txt_clusters_synt_new_un.append(cl1)
            delete_next = True
        else:
            txt_clusters_synt_new_un.append(cl1)
            delete_next = False
    txt_clusters_synt_new_un.append(txt_clusters_synt_new[-1])
    return txt_clusters_synt_new_un


def rewrite_acoustic_markers(best_syntagma_distribution, best_operations, ac_labels, text, txt_clusters):
    new_ac_markers_all = []
    new_txt_distribution = []

    for synt_count in range(len(best_operations)):
        synt_distr = best_syntagma_distribution[synt_count]
        operations = best_operations[synt_count]
        txt_clusters_synt = list(filter(lambda x: synt_distr[0] <= x[0] < synt_distr[1], txt_clusters))

        if len(operations) == 0:
            new_txt_distribution.extend(txt_clusters_synt)
        else:
            for (operation, index, ch) in operations:
                txt_clusters_synt_new = copy(txt_clusters_synt)
                if operation == 'd':
                    to_del_el = txt_clusters_synt[index]
                    txt_clusters_synt_new = txt_clusters_synt_new[:index] + txt_clusters_synt_new[index + 1:]
                    txt_clusters_synt_new[index] = (to_del_el[0], txt_clusters_synt_new[index][1])
                    txt_clusters_synt_new = unite_txt_clusters(txt_clusters_synt_new)
                elif operation == 'r':
                    txt_clusters_synt_new[index] = (txt_clusters_synt_new[index][0], int(ch))
                    txt_clusters_synt_new = unite_txt_clusters(txt_clusters_synt_new)
                elif operation == 'i':
                    txt_clusters_synt_new = txt_clusters_synt_new[:index] + [(txt_clusters_synt_new[index][0], int(ch))] + txt_clusters_synt_new[index:]
                    txt_clusters_synt_new = unite_txt_clusters(txt_clusters_synt_new)

            new_txt_distribution.extend(txt_clusters_synt_new)

    ac_labels_count = 0
    for label in ac_labels:
        if label.level == 'R1' and label.text != "":
            label_text = text[new_txt_distribution[ac_labels_count][0] : new_txt_distribution[ac_labels_count+1][0]]
            ac_labels.append(Label(label.position, "R2", label_text))
            ac_labels_count += 1

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

    best_syntagma_distribution, best_operations_all = make_most_probable_syntagma_distribution_2(ac_string, txt_clusters,
                                                                            word_boundaries_indexes, ac_synt_durations,
                                                                            txt_arr)
    count = 0
    for label in ac_labels:
        if label.level == "Y1" and label.text == "new_synt":
            synt_text = text[best_syntagma_distribution[count][0]: best_syntagma_distribution[count][-1]]
            label.text = synt_text
            count += 1
    
    rewrite_acoustic_markers(best_syntagma_distribution, best_operations_all, ac_labels, text, txt_clusters)
            
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
