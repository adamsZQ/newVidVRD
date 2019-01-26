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
    plt.figure(figsize=(50, 10))
    if bar_type == 0:
        plt.bar(range(len(label_list)), num_list, color='#FAA460', tick_label=label_list)  # sandybrown
        plt.yscale('log')
        plt.tick_params(axis='y', labelsize=20)
        plt.ylabel('Per Category Data Size', fontsize=fontsize+5)
        plt.xticks(rotation=90, fontsize=fontsize)
        plt.gca().yaxis.grid(True)
    elif bar_type == 1:
        plt.barh(range(len(label_list)), num_list, color='rgb', tick_label=label_list)
        plt.xscale('log')
        plt.yticks(fontsize=fontsize)
    # plt.title(title, fontsize=fontsize*1.5)
    plt.axis('tight')
    plt.xlim([-1, len(label_list)])
    plt.tight_layout()
    plt.savefig("{}.jpg".format(title), dpi=400)
    plt.show()


def statistic_visualization_pro_4_rela(label_list, num_list,
                                       highlight=True, title=None):
    fontsize = 20
    plt.figure(figsize=(50, 10))

    if highlight:
        colors = []
        highlight_list = ['next_to', 'in_front_of', 'above', 'beneath', 'behind', 'away', 'towards', 'inside']
        for each_label in label_list:
            if each_label in highlight_list:
                colors.append('#87CEFA')   # lightskyblue
            else:
                colors.append('#FFA07A')   # lightsalmon
        plt.bar(range(len(label_list)), num_list, color=colors, tick_label=label_list)
    else:
        print("No highlight???")

    plt.yscale('log')
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel('Per Category Data Size', fontsize=fontsize+5)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.gca().yaxis.grid(True)
    # plt.title(title, fontsize=fontsize*1.5)
    plt.axis('tight')
    plt.xlim([-1, len(label_list)])
    plt.tight_layout()  # Amazing!!
    plt.savefig("{}.jpg".format(title), dpi=400)
    plt.show()


def get_dataset_visualizations():
    for data_type in ['train']:
        a, b, c, d = get_objects_relations_list(data_type, merge=True)
        statistic_visualization(a, b, data_type + '_objects')
        # statistic_visualization(a, b, data_type + '_objects_1', 1)
        statistic_visualization_pro_4_rela(c, d, title=data_type + '_relations')
        # statistic_visualization(c, d, data_type + '_relations_1', 1)


if __name__ == '__main__':
    get_dataset_visualizations()
    # a, b, c, d = get_objects_relations_list('test')
    # statistic_visualization(a, b)
    # statistic_visualization_pro_4_rela(c, d)
    # for each_type in ['train', 'test']:
    #     a, b, c, d = get_objects_relations_list(each_type)
    #     print(a)
    #     print(len(a))

