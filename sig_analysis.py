import math
from copy import deepcopy

# from scipy import signal as sc_signal
from scipy.fft import rfft, rfftfreq

from utils.signal_classes import Signal, Label


def detect_pauses(signal: Signal, labels: list = [], config: dict = {}):
    config_ = config["pause_detection_parameters"]

    N = int(config_["window_size"] * signal.params.samplerate // signal.params.sampwidth)  # окно анализа 20 мс в отсчетах
    max_signal_ampl = max([abs(i) for i in signal.signal])
    fictive_pause = [0 for i in range(
        int(config_["min_pause_duration"] * signal.params.samplerate * 2))]
    signal_ = fictive_pause + list(signal.signal)
    for i in range(len(signal_) // N):
        signal_part = signal_[i * N: i * N + N]
        max_part_ampl = max([abs(i) for i in signal_part])
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
    # sig_wind = signal * sc_signal.windows.hamming(num_samples)
    yf = rfft(signal)
    xf = rfftfreq(num_samples, 1 / samplerate)
    yf_mod = [i.real for i in yf]

    distribution_dict = {}

    for i in range(int(max(xf) // 250) + 1):         # считаем суммарную плотность на каждые 500 Гц до максимальной частоты в спектре
        distribution_dict[i] = []

    for i in range(len(xf)):
        distribution_dict[int(xf[i] // 250)].append(yf_mod[i])

    for key, value in distribution_dict.items():
        density = sum(list(map(lambda n: abs(n), value)))
        density_log = math.log(density, 2) if density != 0.0 else 0
        distribution_dict[key] = round(density_log, 2)

    return distribution_dict


def unite_label_clusters(labels: list[Label]):
    new_labels = [labels[0]]
    for i, label in enumerate(labels[1: -1]):
        if label.text != new_labels[-1].text and labels[i + 1].text == label.text:
            new_labels.append(label)
    return new_labels


def detect_allophone_classes(signal: Signal, labels: list[Label] = [], config: dict = {}):
    config_ = config["allophone_classes_detection_parameters"]
    samplerate = signal.params.samplerate

    new_labels = deepcopy(labels)
    new_labels_clusters = []
    for start, end in zip(labels, labels[1:]):
        if start.text != "pause" and start.text != "begin":
            syntagma = signal.signal[start.position : end.position]
            # synt_spectral_density_distribution = get_spectral_density_distribution(syntagma, samplerate)
            max_syntagma_ampl = max([abs(i) for i in syntagma])
            N = int(config_["window_size"] * signal.params.samplerate // signal.params.sampwidth)

            avg_syntagma_intensity = sum([abs(i) for i in syntagma]) / len(syntagma)

            for i in range(len(syntagma) // N):
                new_label_position = int(i * N) + start.position
                signal_part = syntagma[i * N: (i + 1) * N]

                max_part_ampl = max([abs(i) for i in signal_part])
                sig_part_spectral_density_distribution = get_spectral_density_distribution(signal_part, samplerate)

                # tmp
                sp_density = list(sig_part_spectral_density_distribution.values())
                less_250_dens = round(sp_density[0], 2)
                dens_250_to_500 = round(sp_density[1], 2)
                dens_500_to_750 = round(sp_density[2], 2)
                dens_750_to_1000 = round(sp_density[3], 2)
                less_500_dens = round(sum(sp_density[:2], 2))
                dens_500_to_1000 = round(sum(sp_density[2:4], 2))
                dens_1000_to_1500 = round(sum(sp_density[4:6], 2))
                less_2500_dens = round(sum(sp_density[:10]), 2)
                dens_2500_to_5000 = round(sum(sp_density[10 : 20]), 2)
                dens_5000_to_7500 = round(sum(sp_density[20: 30]), 2)
                dens_7500_to_10000 = round(sum(sp_density[30:]), 2)
                # first_half_sum = sum(sp_density[: len(sp_density) // 2])
                # sec_half_sum = sum(sp_density[len(sp_density) // 2 : ])
                intensity_by_sample = sum([abs(i) for i in signal_part]) / len(signal_part)
                text_label = f"{round(dens_250_to_500 / dens_500_to_750, 2)} {round(less_250_dens / dens_500_to_750, 2)}"
                if max_part_ampl / max_syntagma_ampl < config_["threshold"]:        # voiceless stops
                    text_label = "stop (voiceless)"
                elif dens_2500_to_5000 > less_2500_dens or dens_5000_to_7500 > less_2500_dens:  # other
                    text_label = "fricative"
                elif intensity_by_sample / avg_syntagma_intensity < 1 or less_250_dens / dens_500_to_750 > 1.2:
                    # low ampl periodic consonants
                    text_label = "other cons"
                else:
                    text_label = "vowel"       # high ampl sonorants and vowels
                new_label = Label(new_label_position, "R1", text_label)
                new_labels_clusters.append(new_label)
                #

    new_labels_clusters = unite_label_clusters(new_labels_clusters)
    new_labels = new_labels + new_labels_clusters

    return new_labels
