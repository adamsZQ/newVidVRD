import os
import json
import ast
import get_data
import sys


def gen_relations():
    relation_res = []

    for path, dir_list, file_list in os.walk(r"../data/train"):
        for file in file_list:
            with open(os.path.join(path, file), 'r') as f:
                train_json = json.loads(f.read())
                for each_ins in train_json["relation_instances"]:
                    relation_res.append(each_ins["predicate"])

    relation_set = set(relation_res)
    relation_first = []
    relation_second = []
    for each_re in relation_set:
        if '_' in each_re:
            each_re_list = each_re.split('_')
            relation_first.append(each_re_list[0])
            relation_second.append(each_re_list[1])
        else:
            relation_first.append(each_re)
    relation_first_set = set(relation_first)
    relation_second_set = set(relation_second)
    i = 0
    first_dict = {}
    for re_fir in relation_first_set:
        first_dict.setdefault(re_fir, i)
        i += 1

    j = 0
    sec_dict = {}
    for re_sec in relation_second_set:
        sec_dict.setdefault(re_sec, j)
        j += 1

    return first_dict, sec_dict


def store_relation_dict():
    first_dict, second_dict = gen_relations()
    with open('../data/first_relation_dict.txt', 'w+') as first_file:
        first_file.write(str(first_dict))

    with open('../data/second_relation_dict.txt', 'w+') as second_file:
        second_file.write(str(second_dict))


def get_vvrd_truth_list(data_type, relation_type):
    pos = 0 if relation_type == 'first' else 1

    file_path = '../data/vvrd_truth_' + data_type + '_' + relation_type + '_list.json'
    vrd_json = {}
    with open(file_path, 'w+') as f:
        for each_vrd in get_data.gen_vrd_instance(data_type):
            if pos == 1:
                if len(each_vrd.predicate.split('_')) > 1:
                    rel = load_relation(relation_type)[each_vrd.predicate.split('_')[pos]]
                else:
                    rel = -1
            else:
                rel = load_relation(relation_type)[each_vrd.predicate.split('_')[pos]]
            vrd_json[each_vrd.video_id + '_'
                     + str(each_vrd.begin_fid) + '_'
                     + str(each_vrd.end_fid)] = rel
            # each_vrd_json = {each_vrd.video_id + '_'
            #                  + str(each_vrd.begin_fid) + '_'
            #                  + str(each_vrd.end_fid): rel}
        f.write(json.dumps(vrd_json))
    return 'Successfully get VVRD Truth List: ' + file_path


def get_vvrd_truth(data_type, relation_type, video_id, begin_fid, end_fid):
    # datatype: ['train', 'test']
    # relation_type: ['first', 'second']
    vvrd_truth_list_base_path = '../data/vvrd_truth_' + data_type + '_' + relation_type + '_list.json'
    print(vvrd_truth_list_base_path)
    with open(vvrd_truth_list_base_path, 'r') as f:
        vvrd_truth_list = json.loads(f.read())
    return vvrd_truth_list[video_id + '_' + str(begin_fid) + '_' + str(end_fid)]


def load_relation(relation_type):
    relation_type = '../data/' + relation_type + '_relation_dict.txt'
    s = ''
    with open(relation_type, 'r') as f:
        s += f.readline()
    return ast.literal_eval(s)


if __name__ == '__main__':
    for rel_type in ['first', 'second']:
        for train_type in ['train', 'test']:
            get_vvrd_truth_list(train_type, rel_type)
    # vrd_ins = get_data.gen_vrd_instance('train')[1]
    # print(get_vvrd_truth('train', 'first', vrd_ins.video_id, vrd_ins.begin_fid, vrd_ins.end_fid))
