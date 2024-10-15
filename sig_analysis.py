from copy import deepcopy

import numpy as np
from scipy.fft import rfft, rfftfreq

from utils.signal_classes import Signal, Label


def detect_pauses(signal: Signal, labels: list = [], config: dict = {}):
    config_ = config["pause_detection_parameters"]

    N = int(config_["window_size"] * signal.params.samplerate // signal.params.sampwidth)  # окно анализа 20 мс в отсчетах
    max_signal_ampl = max([np.abs(i) for i in signal.signal])
    fictive_pause = [0 for i in range(
        int(config_["min_pause_duration"] * signal.params.samplerate * 2))]
    signal_ = fictive_pause + list(signal.signal)
    for i in range(len(signal_) // N):
        signal_part = signal_[i * N: i * N + N]
        max_part_ampl = max([np.abs(i) for i in signal_part])
        new_label_position = int(i * N - len(fictive_pause))
        to_prev_label_time_distance = (new_label_position - labels[-1].position) / signal.params.samplerate
        if max_part_ampl < config_["threshold"] * max_signal_ampl:
            if labels[-1].text != 'pause':
                if (to_prev_label_time_distance > config_["min_alloph_duration"]
                        or labels[-1].text == 'begin'):
                    labels.append(Label(new_label_position, "Y1", 'pause'))
                elif len(labels) > 2:
                    labels.pop(len(labels) - 1)
        elif labels[-1].text != 'new_synt':
            if (to_prev_label_time_distance > config_["min_pause_duration"]
                    or labels[-1].text == 'begin'):
                labels.append(Label(new_label_position, "Y1", 'new_synt'))
            elif len(labels) > 2:
                labels.pop(len(labels) - 1)
    return labels


def get_spectral_density_distribution(signal, samplerate) -> dict:
    num_samples = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(num_samples, 1 / samplerate)
    yf_mod = [i.real for i in yf]

    distribution_dict = {}

    for i in range(int(max(xf) // 500) + 1):         # считаем суммарную плотность на каждые 500 Гц до максимальной частоты в спектре
        distribution_dict[i] = []

    for i in range(len(xf)):
        distribution_dict[int(xf[i] // 500)].append(yf_mod[i])

    for key, value in distribution_dict.items():
        density = np.log2(sum(list(map(lambda n: np.abs(n), value))) / len(value))
        distribution_dict[key] = round(density, 1)

    return distribution_dict


def detect_allophone_classes(signal: Signal, labels: list[Label] = [], config: dict = {}):
    config_ = config["allophone_classes_detection_parameters"]
    samplerate = signal.params.samplerate

    new_labels = deepcopy(labels)
    for start, end in zip(labels, labels[1:]):
        if start.text == "pause":
            continue

        syntagma = signal.signal[start.position : end.position]
        synt_spectral_density_distribution = get_spectral_density_distribution(syntagma, samplerate)
        max_syntagma_ampl = max([np.abs(i) for i in syntagma])
        N = int(config_["window_size"] * signal.params.samplerate // signal.params.sampwidth)

        for i in range(len(syntagma) // N):
            new_label_position = int(i * N) + start.position
            signal_part = syntagma[i * N: i * N + N]
            sig_part_spectral_density_distribution = get_spectral_density_distribution(signal_part, samplerate)

            # tmp
            text_label = " ".join([str(x) for x in sig_part_spectral_density_distribution.values()])
            new_label = Label(new_label_position, "R1", text_label)
            new_labels.append(new_label)
            #



    return new_labels
