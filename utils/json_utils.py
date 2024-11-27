import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def write_object_to_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NpEncoder)


def get_object_from_json(filename):
    with open(filename) as f:
        obj = json.load(f)
    return obj
