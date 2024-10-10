from cProfile import label

import numpy as np
from scipy.fft import rfft

from utils.signal_classes import Signal, Label, level2code


def detect_pauses(signal: Signal, labels: list = []):
    #TODO: проверить согласные после паузы в дебаге
    N = int(0.02 * signal.params.samplerate // signal.params.sampwidth) # окно анализа 20 мс в отсчетах
    max_signal_ampl = max([np.abs(i) for i in signal.signal])
    for i in range (len(signal.signal) // N):
        signal_part = signal.signal[i * N : i * N + N]
        max_part_ampl = max([np.abs(i) for i in signal_part])
        new_label_position = int(i * N)
        to_prev_label_time_distance = (new_label_position - labels[-1].position) / signal.params.samplerate
        if max_part_ampl < 0.07 * max_signal_ampl:
            if labels[-1].text != 'pause':
                if to_prev_label_time_distance > 0.15 or labels[-1].text == 'begin':
                    labels.append(Label(new_label_position, "Y1", 'pause'))
                elif len(labels) > 2:
                    labels.pop(len(labels) - 1)
        elif labels[-1].text != 'new_synt':
            if to_prev_label_time_distance > 0.15 or labels[-1].text == 'begin':
                labels.append(Label(new_label_position, "Y1", 'new_synt'))
            elif len(labels) > 2:
                labels.pop(len(labels) - 1)
    return labels


def detect_fricative_parts(signal: Signal, labels: list = []):
    pass