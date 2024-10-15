import numpy as np

from utils.signal_classes import Signal, Label


def detect_pauses(signal: Signal, labels: list = [], config: dict = {}):
    N = int(config["pause_detection_parameters"]["window_size"] * signal.params.samplerate // signal.params.sampwidth)  # окно анализа 20 мс в отсчетах
    max_signal_ampl = max([np.abs(i) for i in signal.signal])
    fictive_pause = [0 for i in range(
        int(config["pause_detection_parameters"]["min_pause_duration"] * signal.params.samplerate * 2))]
    signal_ = fictive_pause + list(signal.signal)
    for i in range(len(signal_) // N):
        signal_part = signal_[i * N: i * N + N]
        max_part_ampl = max([np.abs(i) for i in signal_part])
        new_label_position = int(i * N - len(fictive_pause))
        to_prev_label_time_distance = (new_label_position - labels[-1].position) / signal.params.samplerate
        if max_part_ampl < config["pause_detection_parameters"]["threshold"] * max_signal_ampl:
            if labels[-1].text != 'pause':
                if (to_prev_label_time_distance > config["pause_detection_parameters"]["min_alloph_duration"]
                        or labels[-1].text == 'begin'):
                    labels.append(Label(new_label_position, "Y1", 'pause'))
                elif len(labels) > 2:
                    labels.pop(len(labels) - 1)
        elif labels[-1].text != 'new_synt':
            if (to_prev_label_time_distance > config["pause_detection_parameters"]["min_pause_duration"]
                    or labels[-1].text == 'begin'):
                labels.append(Label(new_label_position, "Y1", 'new_synt'))
            elif len(labels) > 2:
                labels.pop(len(labels) - 1)
    return labels


def detect_fricative_parts(signal: Signal, labels: list = [], config: dict = {}):
    return labels
