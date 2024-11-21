import glob

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate, find_peaks

from sig_analysis import get_spectral_density_distribution, get_zero_cross_rate
from utils.signal_classes import Seg, Signal


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
            peaks = find_peaks(yf_mod_log, height=17)
            number_of_peaks_before_5000 = len([x for x in peaks[0] if x < len(yf_mod_log) / 2])
            number_of_peaks_after_5000 = len([x for x in peaks[0] if x >= len(yf_mod_log) / 2])
            features_dict[start.text][-1].append(number_of_peaks_before_5000)
            features_dict[start.text][-1].append(number_of_peaks_after_5000)

            # 4. spectral density distribution
            sig_part_spectral_density_distribution = get_spectral_density_distribution(signal_part,
                                                                                       signal_.params.samplerate)
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

            features_dict[start.text][-1].append([less_500_dens, dens_500_to_1000, dens_1000_to_1500, dens_1500_to_2000])
            features_dict[start.text][-1].append([less_2500_dens, dens_2500_to_5000, dens_5000_to_7500, dens_7500_to_10000])

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

    return allophone_stat

seg_b1 = Seg(r"D:\pycharm_projects\word_segmentator\test_data\source_data\cta0004.seg")
signal_ = Signal(r"D:\pycharm_projects\word_segmentator\test_data\source_data\cta0004.sbl")
fld_name = r"D:\corpora\corpres\cta"
window_size = 0.04

# features_dict = get_statistics_from_b1(seg_b1, signal_, 0.04)


get_allophone_statistics_for_corpus(fld_name, window_size)


# vowels_ = tuple(['a', 'o', 'e', 'u', 'y', 'i'])
# sonorants_ = tuple(['n', 'm', 'l', 'r', 'v', 'j'])
#
# vowels = [key for key in list(features_dict.keys()) if key.startswith(vowels_)]
# sonorarnts = [key for key in list(features_dict.keys()) if key.startswith(sonorants_)]
# cons = [key for key in list(features_dict.keys()) if
#         not key.startswith(vowels_) and not key.startswith(sonorants_) and key != ""]
#
# for key, value in features_dict.items():
#     if key in vowels:
#         first_5000 = np.array([sum(x[1: 21]) for x in value])
#         sec_5000 = np.array([sum(x[21: ]) for x in value])
#         print("mean, max, min 1000Hz / 2000Hz spect density of ", key, round(np.mean(first_5000 / sec_5000), 2),
#               round(max(first_5000 / sec_5000), 2), round(min(first_5000 / sec_5000), 2))
#
# print('\n\n')
#
# for key, value in features_dict.items():
#     if key in sonorarnts:
#         first_1000 = np.array([sum(x[1: 21]) for x in value])
#         sec_1000 = np.array([sum(x[21: ]) for x in value])
#         print("mean, max, min 1000Hz / 2000Hz spect density of ", key, round(np.mean(first_1000 / sec_1000), 2),
#               round(max(first_1000 / sec_1000), 2), round(min(first_1000 / sec_1000), 2))
#
# print('\n\n')
#
# for key, value in features_dict.items():
#     if key in cons:
#         first_1000 = np.array([sum(x[1: 21]) for x in value])
#         sec_1000 = np.array([sum(x[21: ]) for x in value])
#         print("mean, max, min 1000Hz / 2000Hz spect density of ", key, round(np.mean(first_1000 / sec_1000), 2),
#               round(max(first_1000 / sec_1000), 2), round(min(first_1000 / sec_1000), 2))
#
# print('\n\n')
#
# pause_intensities = [x[0] for x in features_dict[""]]
# print("mean, max, min intensity of pause", np.mean(pause_intensities), max(pause_intensities), min(pause_intensities))
