import json
import os
import pickle
import numpy as np
import VRDInstance
from extract_features import extract_split_features

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


def gen_vrd_instance(feature_type):
    root_ins_path = '../data/VidVRD-features'

    vrd_list = []
    if feature_type == 'train':
        ins_dir = os.path.join(root_ins_path, 'train_Instances')

    else:
        ins_dir = os.path.join(root_ins_path, 'test_Instances')

    if not os.path.exists(ins_dir):
        os.makedirs(ins_dir)

        if feature_type == 'train':
            json_list = get_train_data()
        else:
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

        with open(ins_dir + '/instances.pkl', 'wb+') as f:
            pickle.dump(vrd_list, f)

    else:
        with open(ins_dir + '/instances.pkl', 'rb') as f:
            vrd_list = pickle.load(f)

    return vrd_list


def get_vid_sep_dict():
    train_video_list = []
    test_video_list = []
    train_each_video_dict = {}
    test_each_video_dict = {}
    file_path = '../data/VidVRD-features/separate_features/'
    if not os.path.exists(file_path + 'train_list.json') or os.path.exists(file_path + 'test_list.json'):
        for feature_type in ['train', 'test']:
            vrd_list = gen_vrd_instance(feature_type)
            if feature_type == 'train':
                for each_vrd in vrd_list:
                    if each_vrd.video_id not in train_video_list:
                        train_video_list.append(each_vrd.video_id)
                        train_each_video_dict[each_vrd.video_id] = []
                        train_each_video_dict[each_vrd.video_id].append((each_vrd.begin_fid, each_vrd.end_fid))
                    else:
                        if (each_vrd.begin_fid, each_vrd.end_fid) not in train_each_video_dict[each_vrd.video_id]:
                            train_each_video_dict[each_vrd.video_id].append((each_vrd.begin_fid, each_vrd.end_fid))
                with open(file_path + 'train_list.json', 'w+') as train_f:
                    train_f.write(str(train_each_video_dict))
            else:
                for each_vrd in vrd_list:
                    if each_vrd.video_id not in test_video_list:
                        test_video_list.append(each_vrd.video_id)
                        test_each_video_dict[each_vrd.video_id] = []
                        test_each_video_dict[each_vrd.video_id].append((each_vrd.begin_fid, each_vrd.end_fid))
                    else:
                        if (each_vrd.begin_fid, each_vrd.end_fid) not in test_each_video_dict[each_vrd.video_id]:
                            test_each_video_dict[each_vrd.video_id].append((each_vrd.begin_fid, each_vrd.end_fid))
                with open(file_path + 'test_list.json', 'w+') as test_f:
                    test_f.write(str(test_each_video_dict))
    else:
        with open(file_path + 'train_list.json', 'r') as f:
            train_each_video_dict = json.loads(f.read())
        with open(file_path + 'test_list.json', 'r') as f:
            test_each_video_dict = json.loads(f.read())
    return train_each_video_dict, test_each_video_dict


def gen_vrd_feature(output_dir='../data/VidVRD-features/separate_features'):
    base_vid_path = '../data/VidVRD-videos/'

    for feature_type in ['train', 'test']:
        if feature_type == 'train':
            video_dict, _ = get_vid_sep_dict()
        else:
            _, video_dict = get_vid_sep_dict()

        output_dir = os.path.join(output_dir, feature_type)
        for directory in [output_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        for each_vrd_id in video_dict.keys():
            # input_vid, output_dir, begin_fid, end_fid

            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n"
                  + "now is extracting: " + each_vrd_id)
            for (begin_fid, end_fid) in video_dict[each_vrd_id]:
                print('( ' + str(begin_fid) + ', ' + str(end_fid) + ' )')
                extract_split_features(
                    base_vid_path + each_vrd_id + '.mp4',
                    output_dir,
                    begin_fid,
                    end_fid
                )


if __name__ == '__main__':
    # get_vid_sep_dict()
    # gen_vrd_feature()
    print(gen_vrd_instance('train')[10].predicate)
