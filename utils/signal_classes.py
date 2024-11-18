import wave
import struct


def detect_encoding(file_path):
    encoding = "utf-8"
    try:
        l = open(file_path, 'r', encoding="utf-8").read()
        if l.startswith("\ufeff"):  # byte order mark
            encoding = "utf-8-sig"
    except UnicodeDecodeError:
        try:
            open(file_path, 'r', encoding="utf-16").read()
            encoding = "utf-16"
        except UnicodeError:
            encoding = "cp1251"
    return encoding


from itertools import product

letters = "GBRY"
nums = "1234"
levels = [ch + num for num, ch in product(nums, letters)]
level_codes = [2 ** i for i in range(len(levels))]

level2code = {i: j for i, j in zip(levels, level_codes)}
code2level = {j: i for i, j in zip(levels, level_codes)}

sampwidth_to_char = {1: "c", 2: "h", 4: "i"}


class Params():
    def __init__(self, srate, swidth, n_channels) -> None:
        self.samplerate = srate
        self.sampwidth = swidth
        self.numchannels = n_channels


class Label():
    def __init__(self, position, level, text):
        self.position = position
        self.level = level
        self.text = text


class Seg:
    def __init__(self, filename: str = None, labels: list = [], params: Params = Params(22050, 2, 1)):
        self.filename = filename
        self.labels = labels
        self.params = params

    def read_seg_file(self):
        try:
            with open(self.filename, "r", encoding=detect_encoding(self.filename)) as f:
                lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(self.filename, " не найден")

        self.init_params()

        try:
            index_labels = lines.index('[LABELS]')
        except ValueError:
            print("Seg-файл не содержит секции LABELS")

        labels_ = lines[index_labels + 1:]
        labels_arr = [Label(
            int(line.split(",")[0]) // self.params.sampwidth // self.params.numchannels,
            code2level[int(line.split(",")[1])],
            line.split(",")[2]
        ) for line in labels_ if line.count(",") >= 2]

        self.labels = labels_arr

    def init_params(self):
        try:
            with open(self.filename, "r", encoding=detect_encoding(self.filename)) as f:
                lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(self.filename, " не найден")

        try:
            index_params = lines.index('[PARAMETERS]')
        except ValueError:
            print("Seg-файл не содержит секции PARAMETERS")

        try:
            index_labels = lines.index('[LABELS]')
        except ValueError:
            print("Seg-файл не содержит секции LABELS")

        parameters = lines[index_params + 1: index_labels]

        param_dict = {str(line.split("=")[0]): int(line.split("=")[1]) for line in parameters}

        self.params = Params(param_dict["SAMPLING_FREQ"], param_dict["BYTE_PER_SAMPLE"], param_dict["N_CHANNEL"])

    def write_seg_file(self):
        params = {
            "SAMPLING_FREQ": self.params.samplerate,
            "BYTE_PER_SAMPLE": self.params.sampwidth,
            "CODE": 0,
            "N_CHANNEL": self.params.numchannels,
            "N_LABEL": len(self.labels)
        }
        with open(self.filename, "w", encoding="utf-8-sig") as f:
            f.write("[PARAMETERS]\n")
            for key in params.keys():
                f.write(key + '=' + str(params[key]) + "\n")
            f.write("[LABELS]\n")
            for label in self.labels:
                pos = label.position * self.params.numchannels * self.params.sampwidth
                f.write(f"{pos}, {level2code[label.level]}, {label.text}\n")
        print("Параметры и метки записаны в файл ", self.filename)

        def get_labels_in_pairs(self, num_samples):
            ends = [end.position for start, end in zip(self.labels, self.labels[1:])]
            ends.append(num_samples)
            return [(label.position, ends[i], label.text) for i, label in enumerate(self.labels)]


class Signal:
    def __init__(self, filename: str, signal: list = [], params: Params = None, seg: Seg = None):
        self.signal: list = signal
        self.filename: str = filename
        self.seg: Seg = seg
        self.params: Params = params

    def init_params(self):
        if self.params is not None:
            return
        if self.seg is not None:
            self.seg.init_params()
            self.params = self.seg.params
        else:
            default_params = Params(22050, 2, 1)
            self.params = default_params

    def read_wav(self):
        try:
            f = wave.open(self.filename)
        except FileNotFoundError:
            print(self.filename, " не найден")

        num_samples = f.getnframes()
        samplerate = f.getframerate()
        sampwidth = f.getsampwidth()
        num_channels = f.getnchannels()

        sampwidth_to_char = {1: "c", 2: "h", 4: "i"}
        fmt = str(num_samples * num_channels) + sampwidth_to_char[sampwidth]

        signal = struct.unpack(fmt, f.readframes(num_samples * num_channels))
        self.signal = signal
        new_params = Params(samplerate, sampwidth, num_channels)
        self.params = new_params

    def read_sbl(self):
        with open(self.filename, "rb") as f:
            raw_signal = f.read()
        num_samples = len(raw_signal) // self.params.sampwidth
        fmt = str(num_samples) + sampwidth_to_char[self.params.sampwidth]
        signal = struct.unpack(fmt, raw_signal)
        self.signal = signal

    def read_sound_file(self):
        self.init_params()
        if self.filename.endswith(".wav"):
            self.read_wav()
        elif self.filename.endswith(".sbl"):
            self.read_sbl()
        else:
            raise ValueError("Неизвестное расширение, ", self.filename)

    def write_wav_file(self):
        num_samples = self.params.samplerate * 2
        sampwidth_to_char = {1: "c", 2: "h", 4: "i"}
        fmt = str(num_samples) + sampwidth_to_char[self.params.sampwidth]

        signal_ = struct.pack(fmt, *self.signal)

        f = wave.open(self.filename, "wb")
        f.setnchannels(self.params.numchannels)
        f.setsampwidth(self.params.sampwidth)
        f.setframerate(self.params.samplerate)
        f.writeframes(signal_)
        f.close()
