from matplotlib import pyplot as plt
import numpy as np

from scipy import signal
from scipy.fft import fft, fftfreq

from utils.signal_classes import Signal

new_signal = Signal(r"D:\pycharm_projects\word_segmentator\test_data\lin_mod_windows_hw.wav")
new_signal.read_sound_file()

sig = new_signal.signal[len(new_signal.signal) // 2 - 256 : len(new_signal.signal) // 2 + 256]

num_samples = len(sig)
sample_rate = new_signal.params.samplerate

yf = fft(sig) # комплексный спектр
xf = fftfreq(num_samples, 1 / sample_rate) # частоты, аргументы - число отсчетов и период дискретизации (в секундах)
yf_mod = abs(yf) # действительная часть комплексного спетктра

hamming_w = signal.get_window("hamming", num_samples)
windowed_signal_hamming = sig * hamming_w

yf_windowed = fft(windowed_signal_hamming) / 0.54   # не забываем поделить на норму окна
xf = fftfreq(num_samples, 1 / sample_rate)

plt.plot(xf, np.abs(yf_windowed))
plt.xlabel("Frequency, Hz")
plt.ylabel("Amplitude")
plt.axis([0, 1500, 0, max(yf_windowed) * 6])
plt.grid()
plt.show()