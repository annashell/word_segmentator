import json


def write_object_to_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def get_object_from_json(filename):
    with open(filename) as f:
        obj = json.load(f)
    return obj
