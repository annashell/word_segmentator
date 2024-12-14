import glob
import os
from functools import cache

import whisper


@cache
def load_model(model_type):
    return whisper.load_model(model_type)


def transcribe_from_folder(fld_name):
    model = load_model("small")

    wav_files = glob.glob(f"{fld_name}/*.wav", recursive=True)

    for file in wav_files:
        result = model.transcribe(file, language='ru')
        new_filename = os.path.splitext(file)[0] + ".txt"
        with open(new_filename, "w") as f:
            f.write(result['text'].strip())


folder_name = r"D:\test_andre"

transcribe_from_folder(folder_name)
