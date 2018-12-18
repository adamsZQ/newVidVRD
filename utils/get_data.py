import json
import os
import numpy as np
import VRDInstance

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


def get_frames(frames_path):
    frames_path_res = []
    for path, dir_list, file_list in os.walk(frames_path):
        for file in file_list:
            frames_path_res.append(os.path.join(path, file))
    return frames_path_res


def load_feature(file_path):
    return np.load(file_path)


def gen_feature(feature_type):
    json_list = []
    vrd_list = []
    if feature_type == 'train':
        json_list = get_train_data()
    elif feature_type == 'test':
        json_list = get_test_data()

    for each_json in json_list:
        video_id = each_json['video_id']
        objs = {}
        for each_obj in each_json['subject/objects']:
            objs[each_obj['tid']] = each_obj['category']
        for each_relation in each_json['relation_instances']:
            # video_id, objects, begin_fid, end_fid, subject_tid, object_tid, predicate
            vrd_ins = VRDInstance.VRDInstance(
                video_id,
                objs,
                each_relation['begin_fid'],
                each_relation['end_fid'],
                each_relation['subject_tid'],
                each_relation['object_tid'],
                each_relation['predicate']
            )
            vrd_list.append(vrd_ins)
    return vrd_list


if __name__ == '__main__':
    for each in gen_feature('test'):
        print(each.predicate)
