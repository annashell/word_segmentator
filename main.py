from scipy.special import other

from sig_analysis import detect_pauses, detect_fricative_parts
from utils.signal_classes import Signal, Seg


def process_signal(signal: Signal) -> list:
    """
    detects cluster boundaries in audio
    writes seg-file with word boundaries labels
    """
    labels = []
    labels = detect_pauses(signal, labels)
    labels = detect_fricative_parts(signal, labels)

    return labels


def process_text(text: str) -> (list, list):
    """
    detects cluster boundaries in text
    """
    vowels = ('а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я')
    sonorants = ('й', 'л', 'м', 'н', 'р')
    plosives = ('п', 'б', 'т', 'д', 'к', 'г')
    other = ('в', 'ж', 'з', 'с', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь')
    clusters = []
    word_boundaries_indexes = []
    return clusters, word_boundaries_indexes


def make_alignment(signal_clusters, text_clusters, word_boundaries_indexes) -> list:
    """
    makes text and audio labels alignment
    :return list of dicts, where key is a word, value - start and end of the word in audio
    """
    word_boundaries = []
    return word_boundaries


def main(wav_fn, text) -> None:
    """
    writes seg-file with word boundaries
    :param wav_fn: audio filename
    :param text: correspondent text
    """
    signal = Signal(wav_fn)
    ac_labels = process_signal(signal)
    txt_clusters, word_boundaries_indexes = process_text(text)
    word_boundaries = make_alignment(ac_labels, txt_clusters, word_boundaries_indexes)

    new_seg_fn = wav_fn.split(".")[0] + ".seg_Y1"
    new_seg = Seg(new_seg_fn, word_boundaries, signal.params)
    new_seg.write_seg_file()
    print(f"Границы слов записаны в файл {new_seg_fn}")
