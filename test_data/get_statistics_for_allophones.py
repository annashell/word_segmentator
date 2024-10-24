import numpy as np

from sig_analysis import get_spectral_density_distribution
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

        avg_alloph_intensity = sum([abs(i) for i in alloph]) / len(alloph)

        features_dict[start.text].append([round(avg_alloph_intensity / avg_signal_intensity, 2)])

        avg_spectral_density = np.zeros(40)

        for i in range(len(alloph) // N):
            signal_part = alloph[i * N: (i + 1) * N]

            sig_part_spectral_density_distribution = get_spectral_density_distribution(signal_part,
                                                                                       signal_.params.samplerate)
            for j, value in enumerate(sig_part_spectral_density_distribution.values()):
                if j < 40:
                    avg_spectral_density[j] += value

        avg_spectral_density = [round(x / (len(alloph) // N), 2) for x in avg_spectral_density]
        features_dict[start.text][-1].extend(avg_spectral_density)

    return features_dict


seg_b1 = Seg(r"D:\pycharm_projects\word_segmentator\test_data\_cta0004.seg_B1")
signal_ = Signal(r"D:\pycharm_projects\word_segmentator\test_data\_cta0004.sbl")

features_dict = get_statistics_from_b1(seg_b1, signal_, 0.02)

vowels_ = tuple(['a', 'o', 'e', 'u', 'y', 'i'])
sonorants_ = tuple(['n', 'm', 'l', 'r', 'v', 'j'])

vowels = [key for key in list(features_dict.keys()) if key.startswith(vowels_)]
sonorarnts = [key for key in list(features_dict.keys()) if key.startswith(sonorants_)]
cons = [key for key in list(features_dict.keys()) if
        not key.startswith(vowels_) and not key.startswith(sonorants_) and key != ""]

for key, value in features_dict.items():
    if key in vowels:
        first_5000 = np.array([sum(x[1: 21]) for x in value])
        sec_5000 = np.array([sum(x[21: ]) for x in value])
        print("mean, max, min 1000Hz / 2000Hz spect density of ", key, round(np.mean(first_5000 / sec_5000), 2),
              round(max(first_5000 / sec_5000), 2), round(min(first_5000 / sec_5000), 2))

print('\n\n')

for key, value in features_dict.items():
    if key in sonorarnts:
        first_1000 = np.array([sum(x[1: 21]) for x in value])
        sec_1000 = np.array([sum(x[21: ]) for x in value])
        print("mean, max, min 1000Hz / 2000Hz spect density of ", key, round(np.mean(first_1000 / sec_1000), 2),
              round(max(first_1000 / sec_1000), 2), round(min(first_1000 / sec_1000), 2))

print('\n\n')

for key, value in features_dict.items():
    if key in cons:
        first_1000 = np.array([sum(x[1: 21]) for x in value])
        sec_1000 = np.array([sum(x[21: ]) for x in value])
        print("mean, max, min 1000Hz / 2000Hz spect density of ", key, round(np.mean(first_1000 / sec_1000), 2),
              round(max(first_1000 / sec_1000), 2), round(min(first_1000 / sec_1000), 2))

print('\n\n')

pause_intensities = [x[0] for x in features_dict[""]]
print("mean, max, min intensity of pause", np.mean(pause_intensities), max(pause_intensities), min(pause_intensities))
