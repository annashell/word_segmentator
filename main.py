import re
from difflib import ndiff

from sig_analysis import detect_pauses, detect_allophone_classes
from utils.json_utils import get_object_from_json
from utils.signal_classes import Signal, Seg, Label


def process_signal(signal: Signal, config: dict) -> list:
    """
    detects cluster boundaries in audio
    writes seg-file with word boundaries labels
    """
    labels = [Label(0, "Y1", 'begin')]
    labels = detect_pauses(signal, labels, config)
    labels = detect_allophone_classes(signal, labels, config)

    return labels


char_dict = {
    'а': ['a'],
    'б': ['b', 'p'],
    'в': ['v', 'f'],
    'г': ['g', 'k'],
    'д': ['d', 't', "d'"],
    'е': ['e'],
    'ё': ['o'],
    'ж': ['zh'],
    'з': ['z', 's'],
    'и': ['i'],
    'й': ['j'],
    'к': ['k'],
    'л': ['l'],
    'м': ['m'],
    'н': ['n'],
    'о': ['o'],
    'п': ['p'],
    'р': ['r'],
    'с': ['s'],
    'т': ['t', 't', "t'"],
    'у': ['u'],
    'ф': ['f'],
    'х': ['h'],
    'ц': ['c'],
    'ч': ['ch'],
    'ш': ['sh'],
    'щ': ['sc'],
    'ъ': [''],
    'ы': ['y'],
    'ь': [''],
    'э': ['e'],
    'ю': ['u'],
    'я': ['a'],
}

stop_signs = ('.', ',', ':', ';', '!', '?')
voised_cons = ('б', 'в', 'г', 'д', 'ж', 'з', 'й', 'л', 'м', 'н', 'р')
unvoised_cons = ('к', 'п', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ')
vowels = ('а', 'э', 'о', 'у', 'ы', 'и', 'е', 'ё', 'ю', 'я')
softening_vowels = ('и', 'е', 'ё', 'ю', 'я')


def translate_to_latin(text: str) -> str:
    latin_text = "0"

    text_ = text.lower().strip().replace(' ', '0')

    for i, ch in enumerate(text_):
        if ch in vowels:
            latin_text += char_dict[ch][0] + " "
        elif ch in ('т', 'д') and i < len(text_) - 1 and text_[i + 1] in softening_vowels:
            latin_text += char_dict[ch][2] + " "
            continue
        elif ch in ('б', 'в', 'г', 'д', 'з'):
            if i < len(text_) - 1:
                if text_[i + 1] in stop_signs or text_[i + 1] in unvoised_cons or (
                        text_[i + 1] == '0' and text_[i + 2] in unvoised_cons):
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


def process_text(text: str) -> (list, list):
    """
    detects cluster boundaries in text
    0 - vowels and sonorants
    1 - fricative
    2 - voiceless stops
    3 - other consonants
    """
    latin_txt = translate_to_latin(text)  # точка для добавления паузы в начале

    vowels_and_sonorant = ('a', 'e', 'i', 'u', 'o', 'y', 'l', 'm', 'n', 'r', 'v', "l'", "m'", "n'", "r'", "v'")
    voiceless_stops = ('p', 't', 'k', "p'", "k'")
    fricative = ('z', 'z', 'zh', 's', 'f', 'h', "s'", "f'", "h'", 'ch', 'sh', 'sc', "t'", "d'", "c")
    other = ('b', 'd')

    clusters = [(0, 9)]
    word_boundaries_indexes = [0]

    for i, ch in enumerate(latin_txt.lower().strip().split()):
        if ch == '0':
            word_boundaries_indexes.append(i)  # -1 из-за границы слова в начале
        elif ch in vowels_and_sonorant and clusters[-1][1] != 0:
            clusters.append((i, 0))
        elif ch in fricative and clusters[-1][1] != 1:
            clusters.append((i, 1))
        elif ch in voiceless_stops and clusters[-1][1] != 2:
            clusters.append((i, 2))
        elif ch in other and clusters[-1][1] != 3:
            clusters.append((i, 3))

    clusters = clusters[1:]
    return clusters, word_boundaries_indexes


lbl_to_str_dict = {
    "": "",
    "fricative": "1",
    "vowel or sonorant": "0",
    "other cons": "3",
    "stop (voiceless)": "2",
    "begin": "",
    "new_synt": "",
    "pause": "4"
}


def convert_ac_labels_to_string(ac_labels):
    ac_string = "4"
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


def define_syntagmas(ac_labels, txt_clusters):
    pass


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

    # 2. Получение псевдотранскрипции и границ слов
    txt_clusters, word_boundaries_indexes = process_text(text)

    # 3. Сопоставление акустических меток с текстом, поиск пауз
    # ac_labels = define_syntagmas(ac_labels, txt_clusters)
    # acoustic_parts = split_acoustic_labels(ac_labels)
    # text_parts = split_text_clusters(txt_clusters)

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


wav_fn = r"D:\pycharm_projects\word_segmentator\data\source_data\cta0004.wav"
text_fn = r"D:\pycharm_projects\word_segmentator\data\source_data\cta0004.txt"

main(wav_fn, text_fn)
