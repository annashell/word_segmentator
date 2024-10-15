from sig_analysis import detect_pauses, detect_fricative_parts
from utils.json_utils import get_object_from_json
from utils.signal_classes import Signal, Seg, Label


def process_signal(signal: Signal, config: dict) -> list:
    """
    detects cluster boundaries in audio
    writes seg-file with word boundaries labels
    """
    labels = [Label(0, "Y1", 'begin')]
    labels = detect_pauses(signal, labels, config)
    labels = detect_fricative_parts(signal, labels, config)

    return labels


def process_text(text: str) -> (list, list):
    """
    detects cluster boundaries in text
    0 - vowels
    1 - sonorants
    2 - plosives
    3 - other consonants
    4 - pauses
    """
    vowels = ('а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я')
    sonorants = ('й', 'л', 'м', 'н', 'р')
    plosives = ('п', 'б', 'т', 'д', 'к', 'г')
    other = ('в', 'ж', 'з', 'с', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь')

    clusters = [(0, 9)]
    word_boundaries_indexes = []

    for i, ch in enumerate(text.lower().strip()):
        if ch.isspace():
            word_boundaries_indexes.append(i)
        elif ch in ('.', ',', ':', ';'):
            clusters.append((i, 4))
        elif ch in vowels and clusters[-1][1] != 0:
            clusters.append((i, 0))
        elif ch in sonorants and clusters[-1][1] != 1:
            clusters.append((i, 1))
        elif ch in plosives and clusters[-1][1] != 2:
            clusters.append((i, 2))
        elif ch in other and clusters[-1][1] != 3:
            clusters.append((i, 3))

    clusters = clusters[1:]
    return clusters, word_boundaries_indexes


def make_alignment(signal_clusters, text_clusters, word_boundaries_indexes) -> list:
    """
    makes text and audio labels alignment
    :return list of dicts, where key is a word, value - start and end of the word in audio
    """
    word_boundaries = []
    return word_boundaries


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
    ac_labels = process_signal(signal, config)
    txt_clusters, word_boundaries_indexes = process_text(text)
    # word_boundaries = make_alignment(ac_labels, txt_clusters, word_boundaries_indexes)

    new_seg_fn = wav_fn.split(".")[0] + ".seg_Y1"
    # new_seg = Seg(new_seg_fn, word_boundaries, signal.params)
    new_seg = Seg(new_seg_fn, ac_labels, signal.params)
    new_seg.write_seg_file()
    print(f"Границы слов записаны в файл {new_seg_fn}")


wav_fn =  r"D:\pycharm_projects\word_segmentator\test_data\cta0004.wav"
text_fn = r"D:\pycharm_projects\word_segmentator\test_data\cta0004.txt"


main(wav_fn, text_fn)
