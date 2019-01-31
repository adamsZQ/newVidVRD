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


def statistic_visualization(label_list, num_list, highlight=True, title=None, bar_type=0):
    fontsize = 30
    plt.figure(figsize=(50, 15))
    if bar_type == 0:
        if highlight:
            color_list = []
            human_list = ['adult', 'child', 'baby']
            animal_list = ['dog', 'cat', 'bird', 'duck', 'horse', 'fish', 'elephant', 'chicken',
                           'hamster/rat', 'sheep/goat', 'penguin', 'rabbit', 'pig', 'kangaroo', 'cattle/cow', 'turtle',
                           'panda', 'leopard', 'tiger', 'camel', 'lion', 'crab', 'crocodile', 'stingray',
                           'bear', 'snake', 'squirrel']
            object_list = ['toy', 'car', 'chair', 'table', 'cup', 'sofa', 'ball/sports ball', 'bottle',
                           'screen/monitor', 'guitar', 'bicycle', 'backpack', 'baby seat', 'watercraft', 'camera',
                           'handbag',
                           'cellphone', 'laptop', 'stool', 'dish', 'motorcycle', 'bench', 'piano', 'ski',
                           'cake', 'baby walker', 'snowboard', 'bat', 'bus/truck', 'surfboard', 'faucet',
                           'electric fan',
                           'sink', 'aircraft', 'refrigerator', 'skateboard', 'train', 'fruits', 'traffic light',
                           'suitcase',
                           'bread', 'microwave', 'scooter', 'racket', 'oven', 'antelope', 'vegetables', 'toilet',
                           'stop sign', 'frisbee']
            # print(len(human_list))
            # print(len(animal_list))
            # print(len(object_list))
            for each_label in label_list:
                if each_label in human_list:
                    color_list.append('#FFBFDF')   # light purple
                elif each_label in animal_list:
                    color_list.append('#8F8FEF')   # sandybrown
                elif each_label in object_list:
                    color_list.append('#FAA460')   # light pink
                else:
                    print('There isnt a type for: ' + each_label)

            for i in range(len(label_list)):
                # Uniform space format
                if ' ' in label_list[i]:
                    label_list[i] = label_list[i].replace(" ", "_")

            plt.bar(range(len(label_list)), num_list, color=color_list, tick_label=label_list)
            plt.bar(0, 0, color='#FFBFDF', label='Human')
            plt.bar(0, 0, color='#8F8FEF', label='Animal')
            plt.bar(0, 0, color='#FAA460', label='Other')
        else:
            print('no highlight??')
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
    plt.legend(loc="upper right", prop={'size': 30})
    plt.tight_layout()
    plt.savefig("{}.png".format(title), dpi=400)
    plt.show()


def statistic_visualization_pro_4_rela(label_list, num_list,
                                       highlight=True, title=None):
    fontsize = 30
    plt.figure(figsize=(50, 12))
    # print(len(label_list))

    if highlight:
        colors = []
        highlight_list = ['next_to', 'in_front_of', 'above', 'beneath', 'behind', 'away', 'towards', 'inside']
        for each_label in label_list:
            if each_label in highlight_list:
                colors.append('#FFA07A')   # lightskyblue
            else:
                colors.append('#87CEFA')   # lightsalmon
        plt.bar(range(len(label_list)), num_list, color=colors, tick_label=label_list)
        plt.bar(0, 0, color='#FFA07A', label='Spatial Predicate')
        plt.bar(0, 0, color='#87CEFA', label='Action Predicate')
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
    plt.ylim((1, 200000))
    plt.legend(loc="upper right", prop={'size': 30})
    plt.tight_layout()  # Amazing!!
    plt.savefig("{}.png".format(title), dpi=400)
    plt.show()


def get_dataset_visualizations():
    for data_type in ['train']:
        a, b, c, d = get_objects_relations_list(data_type, merge=True)
        statistic_visualization(a, b, title=data_type + '_objects')
        # statistic_visualization(a, b, data_type + '_objects_1', 1)
        statistic_visualization_pro_4_rela(c, d, title=data_type + '_relations')
        # statistic_visualization(c, d, data_type + '_relations_1', 1)


def statistic_objects():
    # ===================== Objects statistic
    a, b, c, d = get_objects_relations_list('train')
    human_list = ['adult', 'child', 'baby']
    animal_list = ['dog', 'cat', 'bird', 'duck', 'horse', 'fish', 'elephant', 'chicken',
                   'hamster/rat', 'sheep/goat', 'penguin', 'rabbit', 'pig', 'kangaroo', 'cattle/cow', 'turtle',
                   'panda', 'leopard', 'tiger', 'camel', 'lion', 'crab', 'crocodile', 'stingray',
                   'bear', 'snake', 'squirrel']
    object_list = ['toy', 'car', 'chair', 'table', 'cup', 'sofa', 'ball/sports ball', 'bottle',
                   'screen/monitor', 'guitar', 'bicycle', 'backpack', 'baby seat', 'watercraft', 'camera',
                   'handbag',
                   'cellphone', 'laptop', 'stool', 'dish', 'motorcycle', 'bench', 'piano', 'ski',
                   'cake', 'baby walker', 'snowboard', 'bat', 'bus/truck', 'surfboard', 'faucet',
                   'electric fan',
                   'sink', 'aircraft', 'refrigerator', 'skateboard', 'train', 'fruits', 'traffic light',
                   'suitcase',
                   'bread', 'microwave', 'scooter', 'racket', 'oven', 'antelope', 'vegetables', 'toilet',
                   'stop sign', 'frisbee']

    human_indexs = []
    object_indexs = []
    animal_indexs = []

    index_id = 0
    for each_obj in a:
        if each_obj in human_list:
            human_indexs.append(index_id)
        elif each_obj in object_list:
            object_indexs.append(index_id)
        elif each_obj in animal_list:
            animal_indexs.append(index_id)
        else:
            print(each_obj + '??????')
        index_id += 1

    # print(str(len(human_indexs)) + ': ' + str(human_indexs))
    # print(str(len(object_indexs)) + ': ' + str(object_indexs))
    # print(str(len(animal_indexs)) + ': ' + str(animal_indexs))

    human_sum = 0
    object_sum = 0
    animal_sum = 0

    for each_human_idx in human_indexs:
        human_sum += b[each_human_idx]

    for each_obj_idx in object_indexs:
        object_sum += b[each_obj_idx]

    for each_animal_idx in animal_indexs:
        animal_sum += b[each_animal_idx]

    print(human_sum)     # 21749
    print(object_sum)    # 13773
    print(animal_sum)    # 3080
    print(human_sum + object_sum + animal_sum)  # 38602


def statistic_relations():
    # ================== Relations statistic
    a, b, c, d = get_objects_relations_list('train')
    spatial_relations_list = ['next_to', 'in_front_of', 'above', 'beneath', 'behind', 'away', 'towards', 'inside']
    spatial_indexs = []
    relations_indexs = []
    index_id = 0
    for each_rel in c:
        if each_rel in spatial_relations_list:
            spatial_indexs.append(index_id)
        else:
            relations_indexs.append(index_id)
        index_id += 1

    # print(str(len(spatial_indexs)) + ': ' + str(spatial_indexs))
    # print(str(len(relations_indexs)) + ': ' + str(relations_indexs))

    spatial_sum = 0
    relation_sum = 0
    for each_spatial_idx in spatial_indexs:
        spatial_sum += d[each_spatial_idx]

    for each_relation_idx in relations_indexs:
        relation_sum += d[each_relation_idx]

    print(spatial_sum)   # 228286
    print(relation_sum)  # 69066
    print(spatial_sum + relation_sum)   # 297352


if __name__ == '__main__':
    get_dataset_visualizations()
    # statistic_visualization(a, b)
    # statistic_visualization_pro_4_rela(c, d)
    # for each_type in ['train', 'test']:
    #     a, b, c, d = get_objects_relations_list(each_type)
    #     print(a)
    #     print(len(a))
    # statistic_relations()
    # statistic_objects()
