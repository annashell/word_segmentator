from copy import deepcopy

import numpy as np
from numpy import mean

from scipy.signal import find_peaks, correlate

from get_statistics_for_allophones import get_spectral_density_distribution, get_zero_cross_rate, define_classes_probabilities_for_window
from utils.signal_classes import Signal, Label, Seg


def get_zeros(wav_signal: list, N_wind):
    sig_len = len(wav_signal)
    res = int(sig_len % (N_wind // 2))
    num_zeros = int(N_wind / 2 - res)
    wav_signal.extend(np.zeros(num_zeros))
    return wav_signal


def detect_pauses_2(filename: str, labels: list = [], config: dict = {}):
    wav_signal = Signal(filename)
    wav_signal.read_wav()

    wav_signal_ = list(wav_signal.signal)

    config_ = config["pause_detection_parameters"]
    N_wind = int(config_["window_size"] * wav_signal.params.samplerate // wav_signal.params.sampwidth)

    # дополняем сигнал нулями
    wav_signal_ = get_zeros(wav_signal_, N_wind)

    number_of_windows = len(wav_signal_) // N_wind

    # 1. Вычисляем интенсивность сигнала
    intensity = []
    for i in range(number_of_windows):
        start = i * N_wind
        end = start + N_wind
        sig_part = np.array(wav_signal_[start:end])

        # Вычисляем среднюю мощность сигнала в окне
        squared_signal = sig_part ** 2
        intensity.append(np.mean(squared_signal))

    print(max(intensity), min(intensity), np.mean(intensity), np.log(min(intensity)) / np.log(max(intensity)))
    max_intensity = max(intensity)

    # 2. Интервалы выше порога определяем как речевые, ниже - как паузы
    threshold = 0.6
    for i, int_wind in enumerate(intensity):
        if np.log(int_wind) / np.log(max_intensity) > threshold:
            labels.append(Label(i * N_wind, "B1", "voiced"))
        else:
            labels.append(Label(i * N_wind, "B1", "pause"))

    # 3. Объединяем речевые и неречевые интервалы
    labels_united = [labels[0]]
    for label1, label2 in zip(labels, labels[1:]):
        if label2.text != label1.text:
            labels_united.append(label2)

    # 4. Удаляем короткие паузы (скорее всего, это смычные согласные)
    # 5. Удаляем короткие звучащие сегменты (это не речевые звуки)
    labels_united_cleared = []
    indexes_to_delete = []
    n = 1
    for label1, label2 in zip(labels_united[1:], labels_united[2:]):
        if ((label1.text == "voiced" and (
                label2.position - label1.position) / wav_signal.params.samplerate < 0.04)  # min_alloph_duration
                or (label1.text == "pause" and (
                        label2.position - label1.position) / wav_signal.params.samplerate < 0.15)):
            if n < len(labels_united) - 1:
                indexes_to_delete.append(n)
                indexes_to_delete.append(n + 1)
        n += 1

    for i, label in enumerate(labels_united):
        if i not in indexes_to_delete:
            labels_united_cleared.append(label)

    # 6. Пишем seg
    new_seg_fn = filename.split(".")[0] + ".seg_B1"
    new_seg = Seg(new_seg_fn, labels_united_cleared, wav_signal.params)
    new_seg.write_seg_file()
    print(f"Границы слов записаны в файл {new_seg_fn}")


# config = get_object_from_json(r"D:\pycharm_projects\word_segmentator\config.json")
# detect_pauses_2(r"D:\pycharm_projects\word_segmentator\data\av15t.wav", [], config)


def detect_pauses(signal: Signal, labels: list = [], config: dict = {}):
    """

    :param signal:
    :param labels:
    :param config:
    :return:
    """
    config_ = config["pause_detection_parameters"]

    N = int(
        config_["window_size"] * signal.params.samplerate // signal.params.sampwidth)  # окно анализа в отсчетах
    max_signal_ampl = max([abs(i) for i in signal.signal])
    fictive_pause = [0 for i in range(
        int(config_["min_pause_duration"] * signal.params.samplerate * 2))]
    signal_ = fictive_pause + list(signal.signal)
    for i in range(len(signal_) // N):
        signal_part = signal_[i * N: i * N + N]
        mean_part_ampl = mean([abs(i) for i in signal_part])
        new_label_position = int(i * N - len(fictive_pause))
        to_prev_label_time_distance = (new_label_position - labels[-1].position) / signal.params.samplerate
        if mean_part_ampl < config_["threshold"] * max_signal_ampl:
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
            syntagma = signal.signal[start.position: end.position]
            # synt_spectral_density_distribution = get_spectral_density_distribution(syntagma, samplerate)
            max_syntagma_ampl = max([abs(i) for i in syntagma])
            N = int(config_["window_size"] * signal.params.samplerate // signal.params.sampwidth)

            avg_syntagma_intensity = sum([abs(i) for i in syntagma]) / len(syntagma)

            new_label_left = Label(0, "R1", "")
            new_labels_clusters.append(new_label_left)

            for i in range(len(syntagma) // N):
                new_label_position = int(i * N) + start.position
                signal_part = syntagma[i * N: (i + 1) * N]

                probabilities, avg_probabilities, prob_by_rel_interval = define_classes_probabilities_for_window(signal_part, samplerate, avg_syntagma_intensity)
                max_rel_interval_probability = max(list(prob_by_rel_interval.values()))
                probable_classes = [key for key in prob_by_rel_interval.keys() if prob_by_rel_interval[key] == max_rel_interval_probability]

                most_probable_ind = list(avg_probabilities.values()).index(max([value for key, value in avg_probabilities.items() if key in probable_classes]))
                most_probable = list(avg_probabilities.keys())[most_probable_ind]

                max_part_ampl = max([abs(i) for i in signal_part])
                sig_part_spectral_density_distribution = get_spectral_density_distribution(signal_part, samplerate)

                zero_crossing_rate = (get_zero_cross_rate(signal_part) / len(signal_part)) * 100

                # tmp
                sp_density = list(sig_part_spectral_density_distribution.values())

                peaks = find_peaks(sp_density, height=17)

                correlation = correlate(signal_part, signal_part)
                corr_zero_crossing = get_zero_cross_rate(correlation)

                number_of_peaks_before_5000 = len([x for x in peaks[0] if x < len(sp_density) / 2])
                number_of_peaks_after_5000 = len([x for x in peaks[0] if x >= len(sp_density) / 2])

                less_250_dens = round(sp_density[0], 2)
                dens_250_to_500 = round(sp_density[1], 2)
                dens_500_to_750 = round(sp_density[2], 2)
                dens_750_to_1000 = round(sp_density[3], 2)
                dens_1000_to_1250 = round(sp_density[3], 2)
                less_500_dens = round(sum(sp_density[:2], 2))
                less_1000_dens = round(sum(sp_density[:4], 2))
                dens_500_to_1000 = round(sum(sp_density[2:4], 2))
                dens_1000_to_1500 = round(sum(sp_density[4:6], 2))
                dens_1500_to_2000 = round(sum(sp_density[6:8], 2))
                dens_2000_to_2500 = round(sum(sp_density[8:10], 2))
                less_2500_dens = round(sum(sp_density[:10]), 2)
                dens_2500_to_5000 = round(sum(sp_density[10: 20]), 2)
                dens_5000_to_7500 = round(sum(sp_density[20: 30]), 2)
                dens_7500_to_10000 = round(sum(sp_density[30: 40]), 2)
                x = round(less_500_dens / dens_500_to_1000, 2)
                y = round(less_1000_dens / (dens_1000_to_1500 + dens_1500_to_2000), 2)
                # first_half_sum = sum(sp_density[: len(sp_density) // 2])
                # sec_half_sum = sum(sp_density[len(sp_density) // 2 : ])
                intensity_by_sample = sum([abs(i) for i in signal_part]) / len(signal_part)
                # vowel_prob = avg_probabilities["vowels"]
                vow_son_prob = avg_probabilities["vowels or sonorants"]
                st_prob = avg_probabilities["voiceless_stops"]
                fr_prob = avg_probabilities["fricative"]
                other_prob = avg_probabilities["other"]
                text_stat = f"v-s {vow_son_prob} st {st_prob} fr {fr_prob} o {other_prob}"
                text_label = most_probable
                if max_part_ampl / max_syntagma_ampl < config_["threshold"]:  # voiceless stops
                    text_label = f"voiceless_stops"
                elif zero_crossing_rate > 30:
                    # text_label = f"fricative {less_2500_dens} {dens_2500_to_5000} {dens_5000_to_7500} {dens_7500_to_10000}"
                    text_label = f"fricative"
                # elif number_of_peaks_before_5000 >= 3 and less_2500_dens > dens_5000_to_7500:
                #     # low vowel or sonorant
                #     # text_label = f"vowel or sonorant {x} {y} {less_250_dens} {dens_250_to_500} {dens_500_to_750} {dens_750_to_1000}"
                #     text_label = f"vowel or sonorant {text_stat}"
                # else:  # else other
                #     # text_label = f"other cons {x} {y} {less_250_dens} {dens_250_to_500} {dens_500_to_750} {dens_750_to_1000}"
                #     text_label = f"other periodic cons {text_stat}"
                new_label = Label(new_label_position, "R1", text_label)
                new_labels_clusters.append(new_label)
            new_label_right = Label(new_label_position + N, "R1", "")
            new_labels_clusters.append(new_label_right)

    new_labels_clusters = unite_label_clusters(new_labels_clusters)
    new_labels = new_labels + new_labels_clusters

    return new_labels
