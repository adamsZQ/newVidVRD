import json
import os

root_path = os.walk(r"../data")


def get_train_data():
    train_json_res = []
    for path, dir_list, file_list in os.walk(r"../data/train"):
        for file in file_list:
            with open(os.path.join(path, file), 'r') as f:
                train_json_res.append(json.loads(f.read()))
    return train_json_res


def get_test_data():
    test_json_res = []
    for path, dir_list, file_list in os.walk(r"../data/test"):
        for file in file_list:
            with open(os.path.join(path, file), 'r') as f:
                test_json_res.append(json.loads(f.read()))
    return test_json_res
