import glob
import math

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate, find_peaks
from scipy.stats import stats

from utils.json_utils import write_object_to_json, get_object_from_json
from utils.signal_classes import Seg, Signal


def get_zero_cross_rate(signal_part):
    zero_crossing_rate = 0
    for el1, el2 in zip(signal_part, signal_part[1:]):
        if el1 * el2 < 0:
            zero_crossing_rate += 1
    return zero_crossing_rate


def get_spectral_density_distribution(signal_, samplerate) -> dict:
    num_samples = len(signal_)
    sig_wind = signal_ * signal.windows.hamming(num_samples)
    yf = rfft(sig_wind) / 0.5
    xf = rfftfreq(num_samples, 1 / samplerate)
    yf_mod = [i.real for i in yf]

    distribution_dict = {}

    for i in range(
            int(max(xf) // 250) + 1):  # считаем суммарную плотность на каждые 250 Гц до максимальной частоты в спектре
        distribution_dict[i] = []

    for i in range(len(xf)):
        distribution_dict[int(xf[i] // 250)].append(yf_mod[i])

    for key, value in distribution_dict.items():
        density = sum(list(map(lambda n: abs(n), value))) ** 2
        density_log = math.log(density) if density != 0.0 else 0
        distribution_dict[key] = round(density_log, 2)

    return distribution_dict


def get_statistics_from_b1(seg_b1: Seg, signal_: Signal, window_size: float):
    seg_b1.read_seg_file()
    signal_.read_sound_file()

    features_dict = {}

    avg_signal_intensity = sum([abs(i) for i in signal_.signal]) / len(signal_.signal)

    for start, end in zip(seg_b1.labels, seg_b1.labels[1:]):
        if start.text not in features_dict.keys():
            features_dict[start.text] = []

        alloph = signal_.signal[start.position: end.position]
        N = int(window_size * signal_.params.samplerate // signal_.params.sampwidth)

        avg_spectral_density = np.zeros(40)

        for i in range(len(alloph) // N):
            signal_part = alloph[i * N: (i + 1) * N]
            features_dict[start.text].append([])

            num_samples = len(signal_part)
            sig_wind = signal_part * signal.windows.hamming(num_samples)
            yf = rfft(sig_wind) / 0.5
            yf_mod_log = [np.log(i.real) for i in yf]

            # 1. zero-crossing rate
            zero_crossing_rate = (get_zero_cross_rate(signal_part) / len(signal_part)) * 100
            features_dict[start.text][-1].append(round(zero_crossing_rate, 2))

            # 2. autocorrelation zero-crossing
            correlation = correlate(signal_part, signal_part)
            corr_zero_crossing = (get_zero_cross_rate(correlation) / len(correlation)) * 100
            features_dict[start.text][-1].append(round(corr_zero_crossing, 2))

            # 3. spectral peaks number
            sig_part_spectral_density_distribution = get_spectral_density_distribution(signal_part,
                                                                                       signal_.params.samplerate)
            sp_density = list(sig_part_spectral_density_distribution.values())
            peaks = find_peaks(sp_density, height=17)
            number_of_peaks_before_5000 = len([x for x in peaks[0] if x < len(sp_density) / 2])
            number_of_peaks_after_5000 = len([x for x in peaks[0] if x >= len(sp_density) / 2])
            features_dict[start.text][-1].append(number_of_peaks_before_5000)
            features_dict[start.text][-1].append(number_of_peaks_after_5000)

            # 4. spectral density distribution
            for j, value in enumerate(sig_part_spectral_density_distribution.values()):
                if j < 40:
                    avg_spectral_density[j] += value
            avg_spectral_density = [round(x / (len(signal_part) // N), 2) for x in avg_spectral_density]

            less_500_dens = round(sum(avg_spectral_density[:2], 2))
            dens_500_to_1000 = round(sum(avg_spectral_density[2:4], 2))
            dens_1000_to_1500 = round(sum(avg_spectral_density[4:6], 2))
            dens_1500_to_2000 = round(sum(avg_spectral_density[6:8], 2))

            less_2500_dens = round(sum(avg_spectral_density[:10]), 2)
            dens_2500_to_5000 = round(sum(avg_spectral_density[10: 20]), 2)
            dens_5000_to_7500 = round(sum(avg_spectral_density[20: 30]), 2)
            dens_7500_to_10000 = round(sum(avg_spectral_density[30: 40]), 2)

            features_dict[start.text][-1].extend(
                [less_500_dens, dens_500_to_1000, dens_1000_to_1500, dens_1500_to_2000])
            features_dict[start.text][-1].extend(
                [less_2500_dens, dens_2500_to_5000, dens_5000_to_7500, dens_7500_to_10000])

            # 5. mean window amplitude
            avg_alloph_window_intensity = sum([abs(i) for i in signal_part]) / len(signal_part)
            features_dict[start.text][-1].append(round(avg_alloph_window_intensity / avg_signal_intensity, 2))

            # 6. max window amplitude
            max_alloph_window_intensity = max([abs(i) for i in signal_part])
            features_dict[start.text][-1].append(round(max_alloph_window_intensity / avg_signal_intensity, 2))

    return features_dict


def get_allophone_statistics_for_corpus(fld_name, window_size):
    allophone_stat = {}

    sbl_files = glob.glob(f"{fld_name}/*/*.sbl", recursive=True)
    seg_files = glob.glob(f"{fld_name}/*/*.seg_B1", recursive=True)

    for i, sbl_file in enumerate(sbl_files):
        sbl_obj = Signal(sbl_file)
        seg_obj = Seg(seg_files[i])
        new_stat = get_statistics_from_b1(seg_obj, sbl_obj, window_size)
        for key, value in new_stat.items():
            if key in allophone_stat.keys():
                allophone_stat[key].extend(value)
            else:
                allophone_stat[key] = [value]

    # TODO найти min, max, mean для признаков, записать в словарь

    stat_distribution = {}

    field_names = ["zero_cr", "autocor_zero_cr", "sp_peaks_num_before_5000", "sp_peaks_num_after_5000",
                   "less_500_dens", "dens_500_to_1000", "dens_1000_to_1500", "dens_1500_to_2000",
                   "less_2500_dens", "dens_2500_to_5000", "dens_5000_to_7500", "dens_7500_to_10000", "avg_intensity",
                   "max_intensity"]

    for key, value in allophone_stat.items():
        stat_distribution[key] = {}
        for j, name in enumerate(field_names):
            stat_distribution[key][name] = [v[j] for v in value[1:]]

    stat_distribution_by_classes = {
        "vowels": {},
        "sonorants": {},
        "voiceless_stops": {},
        "fricative": {},
        "other": {}
    }
    vowels = ('a', 'e', 'i', 'u', 'o', 'y')
    sonorants = ('l', 'm', 'n', 'r', 'v', "l'", "m'", "n'", "r'", "v'", 'j')
    voiceless_stops = ('p', 't', 'k', "p'", "k'")
    fricative = ('z', 'z', 'zh', 's', 'f', 'h', "s'", "f'", "h'", 'ch', 'sh', 'sc', "t'", "d'", "c", "CH")
    other = ('b', 'd', "b'")

    for key, value in allophone_stat.items():
        new_key = "other"
        if key.startswith(vowels):
            new_key = "vowels"
        elif key in sonorants:
            new_key = "sonorants"
        elif key in voiceless_stops:
            new_key = "voiceless_stops"
        elif key in fricative:
            new_key = "fricative"

        for j, name in enumerate(field_names):
            if name not in stat_distribution_by_classes[new_key].keys():
                stat_distribution_by_classes[new_key][name] = [v[j] for v in value[1:]]
            else:
                stat_distribution_by_classes[new_key][name].extend([v[j] for v in value[1:]])

    stat_distrib_histograms_by_classes = {}
    for key, value in stat_distribution_by_classes.items():
        for inner_key, inner_value in stat_distribution_by_classes[key].items():
            if key not in stat_distrib_histograms_by_classes.keys():
                stat_distrib_histograms_by_classes[key] = {}
            stat_distrib_histograms_by_classes[key][inner_key] = np.histogram(
                stat_distribution_by_classes[key][inner_key])

    stat_distrib_histograms_by_allophones = {}
    for key, value in stat_distribution.items():
        for inner_key, inner_value in stat_distribution[key].items():
            if key not in stat_distrib_histograms_by_allophones.keys():
                stat_distrib_histograms_by_allophones[key] = {}
            stat_distrib_histograms_by_allophones[key][inner_key] = np.histogram(stat_distribution[key][inner_key])

    return stat_distrib_histograms_by_classes, stat_distrib_histograms_by_allophones


def write_stat_json(fld_name, window_size):
    stat_distrib_histograms_by_classes, stat_distrib_histograms_by_allophones = get_allophone_statistics_for_corpus(
        fld_name, window_size)

    write_object_to_json(stat_distrib_histograms_by_classes, "data/stats/stat_distrib_histograms_by_allophones.json")
    write_object_to_json(stat_distrib_histograms_by_allophones, "data/stats/stat_distrib_histograms_by_allophones.json")


def define_classes_probabilities_for_window(signal_part):
    classes_stat_json = "data/stats/stat_distrib_histograms_by_classes.json"
    stats = get_object_from_json(classes_stat_json)

    probabilities = {}

    return probabilities


fld_name = r"D:\corpora\corpres\cta"
window_size = 0.04

# write_stat_json(fld_name, window_size)
