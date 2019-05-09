import json

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# matplotlib.rcParams['text.usetex'] = True


def statistic_relations_new():
    with open('train_rela_sum.json', 'r') as in_f:
        relation_dict = json.load(in_f)
    rela_sum = sorted(relation_dict.items(), key=lambda o: o[1], reverse=True)

    rela_labels = []
    rela_sums = []
    for o in rela_sum:
        rela_labels.append(o[0])
        rela_sums.append(o[1])

    return rela_labels, rela_sums


def statistic_objects_new():
    with open('train_objs_sum.json', 'r') as in_f:
        objs_sum = json.load(in_f)
    objs_sum = sorted(objs_sum.items(), key=lambda o: o[1], reverse=True)

    obj_labels = []
    obj_sums = []
    for o in objs_sum:
        obj_labels.append(o[0])
        obj_sums.append(o[1])

    return obj_labels, obj_sums


def get_dataset_visual_new(obj=True, rela=True):
    if obj:
        obj_labels, obj_nums = statistic_objects_new()
        statistic_visualization_4_obj(obj_labels, obj_nums, title='train_objects')

    if rela:
        label_list, num_list = statistic_relations_new()
        statistic_visualization_4_rela(label_list, num_list, title='train_relations')


def statistic_visualization_4_obj(label_list, num_list, highlight=True, title=None, bar_type=0):
    fontsize = 30
    plt.figure(figsize=(50, 12))
    if bar_type == 0:
        if highlight:
            color_list = []
            human_list = ['adult', 'child', 'baby']
            animal_list = ['dog', 'cat', 'bird', 'duck', 'horse', 'fish', 'elephant', 'chicken',
                           'hamster/rat', 'sheep/goat', 'penguin', 'rabbit', 'pig', 'kangaroo', 'cattle/cow', 'turtle',
                           'panda', 'leopard', 'tiger', 'camel', 'lion', 'crab', 'crocodile', 'stingray',
                           'bear', 'snake', 'squirrel']
            object_list = ['toy', 'car', 'chair', 'table', 'cup', 'sofa', 'ball/sports_ball', 'bottle',
                           'screen/monitor', 'guitar', 'bicycle', 'backpack', 'baby_seat', 'watercraft', 'camera',
                           'handbag',
                           'cellphone', 'laptop', 'stool', 'dish', 'motorcycle', 'bench', 'piano', 'ski',
                           'cake', 'baby_walker', 'snowboard', 'bat', 'bus/truck', 'surfboard', 'faucet',
                           'electric_fan',
                           'sink', 'aircraft', 'refrigerator', 'skateboard', 'train', 'fruits', 'traffic_light',
                           'suitcase',
                           'bread', 'microwave', 'scooter', 'racket', 'oven', 'antelope', 'vegetables', 'toilet',
                           'stop_sign', 'frisbee']
            for each_label in label_list:
                if each_label in human_list:
                    color_list.append('#8F8FEF')  # light purple
                elif each_label in animal_list:
                    color_list.append('#FAA460')  # sandybrown
                elif each_label in object_list:
                    color_list.append('#FFBFDF')  # light pink
                else:
                    print('There isnt a type for: ' + each_label)

            for i in range(len(label_list)):
                # Uniform space format
                if ' ' in label_list[i]:
                    label_list[i] = label_list[i].replace(" ", "_")

            plt.bar(range(len(label_list)), num_list, color=color_list, tick_label=label_list)
            plt.bar(0, 0, color='#8F8FEF', label='Human')
            plt.bar(0, 0, color='#FAA460', label='Animal')
            plt.bar(0, 0, color='#FFBFDF', label='Other')
        else:
            print('no highlight??')
            plt.bar(range(len(label_list)), num_list, color='#FAA460', tick_label=label_list)  # sandybrown
        plt.yscale('log')
        plt.tick_params(axis='y', labelsize=20)
        plt.ylabel('Per Category Data Size', fontsize=fontsize + 5)
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
    plt.savefig("{}.pdf".format(title), dpi=400)
    plt.show()


def statistic_visualization_4_rela(label_list, num_list,
                                   highlight=True, title=None):
    fontsize = 30
    plt.figure(figsize=(50, 12))
    # print(len(label_list))

    if highlight:
        colors = []
        highlight_list = ['next_to', 'in_front_of', 'above', 'beneath', 'behind', 'away', 'towards', 'inside']
        for each_label in label_list:
            if each_label in highlight_list:
                colors.append('#FFA07A')  # lightskyblue
            else:
                colors.append('#87CEFA')  # lightsalmon
        plt.bar(range(len(label_list)), num_list, color=colors, tick_label=label_list)
        plt.bar(0, 0, color='#FFA07A', label='Spatial Predicate')
        plt.bar(0, 0, color='#87CEFA', label='Action Predicate')
    else:
        print("No highlight???")

    plt.yscale('log')
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel('Per Category Data Size', fontsize=fontsize + 5)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.gca().yaxis.grid(True)
    # plt.title(title, fontsize=fontsize*1.5)
    plt.axis('tight')
    plt.xlim([-1, len(label_list)])
    plt.ylim((1, 200000))
    plt.legend(loc="upper right", prop={'size': 30})
    plt.tight_layout()  # Amazing!!
    plt.savefig("{}.pdf".format(title), dpi=400)
    plt.show()


if __name__ == '__main__':
    get_dataset_visual_new()
