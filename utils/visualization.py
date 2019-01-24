import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import vord_utils
import VORDInstance
from collections import Counter


# visualization for vord

def get_objects_relations_dict(data_type):
    # statistics for objects & relations
    objects_dict = {}
    relations_dict = {}
    for each_vord in vord_utils.get_vord_instance('json_list_result_' + data_type + '_data.pkl'):
        each_vord_objects_list, each_vord_relations_list = each_vord.get_object_relations_list()
        for each_vord_object in each_vord_objects_list:
            if each_vord_object in objects_dict.keys():
                objects_dict[each_vord_object] += 1
            else:
                objects_dict[each_vord_object] = 1

        for each_vord_relation in each_vord_relations_list:
            if each_vord_relation in relations_dict.keys():
                relations_dict[each_vord_relation] += 1
            else:
                relations_dict[each_vord_relation] = 1
    return objects_dict, relations_dict


def get_objects_relations_list(data_type, merge=True):
    objects_label_list = []
    objects_num_list = []
    relations_label_list = []
    relations_num_list = []
    if data_type == 'train' or data_type == 'val':
        if merge:
            # merge train and val
            train_obj, train_rel = get_objects_relations_dict('train')
            val_obj, val_rel = get_objects_relations_dict('val')
            objects_dict = dict(Counter(train_obj) + Counter(val_obj))
            relations_dict = dict(Counter(train_rel) + Counter(val_rel))
        else:
            objects_dict, relations_dict = get_objects_relations_dict(data_type)
    else:
        objects_dict, relations_dict = get_objects_relations_dict(data_type)

    for each_object in sorted(objects_dict.items(), key=lambda item: item[1], reverse=True):
        objects_label_list.append(each_object[0])
        objects_num_list.append(each_object[1])
    for each_relation in sorted(relations_dict.items(), key=lambda item: item[1], reverse=True):
        relations_label_list.append(each_relation[0])
        relations_num_list.append(each_relation[1])
    return objects_label_list, objects_num_list, relations_label_list, relations_num_list


def statistic_visualization(label_list, num_list, title=None, bar_type=0):
    fontsize = 20
    plt.figure(figsize=(40, 15))
    if bar_type == 0:
        plt.bar(range(len(label_list)), num_list, color='b', tick_label=label_list)
        plt.yscale('log')
        plt.tick_params(axis='y', labelsize=20)
        plt.ylabel('Per Class Data Size', fontsize=fontsize)
        plt.xticks(rotation=90, fontsize=fontsize)
        plt.gca().yaxis.grid(True)
    elif bar_type == 1:
        plt.barh(range(len(label_list)), num_list, color='rgb', tick_label=label_list)
        plt.xscale('log')
        plt.yticks(fontsize=fontsize)
    # plt.title(title, fontsize=fontsize*1.5)
    plt.axis('tight')
    plt.xlim([-1, len(label_list)])
    plt.savefig("{}.jpg".format(title), dpi=400)
    plt.show()


def get_dataset_visualizations():
    for data_type in ['train', 'test']:
        a, b, c, d = get_objects_relations_list(data_type)
        statistic_visualization(a, b, data_type + '_objects_0')
        # statistic_visualization(a, b, data_type + '_objects_1', 1)
        statistic_visualization(c, d, data_type + '_relations_0')
        # statistic_visualization(c, d, data_type + '_relations_1', 1)


if __name__ == '__main__':
    get_dataset_visualizations()
