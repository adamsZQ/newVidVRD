import os
import json
import ast


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


def load_relation(file_path):
    s = ''
    with open(file_path, 'r') as f:
        s += f.readline()
    return ast.literal_eval(s)


if __name__ == '__main__':
    print(len(load_relation('../data/first_relation_dict.txt')))
