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

    for i in range(int(max(xf) // 500) + 1):         # считаем суммарную плотность на каждые 500 Гц до максимальной частоты в спектре
        distribution_dict[i] = []

    for i in range(len(xf)):
        distribution_dict[int(xf[i] // 500)].append(yf_mod[i])

    for key, value in distribution_dict.items():
        density = sum(list(map(lambda n: abs(n), value)))
        density_log = math.log(density, 2) if density != 0.0 else 0
        distribution_dict[key] = round(density_log, 2)

    return distribution_dict


def detect_allophone_classes(signal: Signal, labels: list[Label] = [], config: dict = {}):
    config_ = config["allophone_classes_detection_parameters"]
    samplerate = signal.params.samplerate

    new_labels = deepcopy(labels)
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
                less_500_dens = round(sp_density[0], 2)
                dens_500_to_1000 = round(sp_density[1], 2)
                dens_1000_to_1500 = round(sp_density[2], 2)
                less_2500_dens = round(sum(sp_density[:5]), 2)
                dens_2500_to_5000 = round(sum(sp_density[5 : 10]), 2)
                dens_5000_to_7500 = round(sum(sp_density[10: 15]), 2)
                dens_7500_to_10000 = round(sum(sp_density[15:]), 2)
                # first_half_sum = sum(sp_density[: len(sp_density) // 2])
                # sec_half_sum = sum(sp_density[len(sp_density) // 2 : ])
                intensity_by_sample = sum([abs(i) for i in signal_part]) / len(signal_part)
                text_label = f"{round(intensity_by_sample / avg_syntagma_intensity, 2)} {round(less_500_dens / dens_500_to_1000, 2)} {round(less_500_dens / dens_1000_to_1500, 2)}"
                if max_part_ampl / max_syntagma_ampl < config_["threshold"]:        # voiceless stops
                    text_label = "2"
                elif dens_2500_to_5000 > less_2500_dens or dens_5000_to_7500 > less_2500_dens:  # other
                    text_label = "4"
                elif intensity_by_sample / avg_syntagma_intensity < 1:        # low ampl periodic consonants
                    text_label = "3"
                else:
                    text_label = "0"        # high ampl sonorants and vowels
                new_label = Label(new_label_position, "R1", text_label)
                new_labels.append(new_label)
                #

    return new_labels
