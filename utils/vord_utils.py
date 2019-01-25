import os
import json
import VORDInstance
import argparse
import visualization
import pickle

root_path = "/home/david/PycharmProjects/VVRD_Dataset10k/10kDataSet"
data_type_paths = ['json_list_result_train_data.pkl',
                   'json_list_result_test_data.pkl',
                   'json_list_result_val_data.pkl']


def get_data_list():
    all_file_dict = {}
    for path, dir_list, _ in os.walk(os.path.join(root_path, "nus-vord")):
        for each_dir in dir_list:
            for each_path, _, file_list in os.walk(os.path.join(path, each_dir)):
                for each_file in file_list:
                    all_file_dict[each_file] = os.path.join(each_path, each_file)
    return all_file_dict


def get_json_list(data_type='none', read_json_data=False,
                  save_result=True, save_path='json_list_result',
                  load_from_save=False, load_from_path='json_list_result'):
    """
    data_type = ['train', 'test', 'val']
    read_json_data, read real json data or not, may need long time, return json list,
    save_result, save loading result to make quicker to get result next time,
    save_path, save result path,
    load_from_save, if has saved results, define the load_from_path and get results.
    """

    if load_from_save is False:
        # initiate files
        json_res = []
        if data_type == 'train':
            data_path = os.path.join(root_path, 'train.txt')
        elif data_type == 'test':
            data_path = os.path.join(root_path, 'test.txt')
        elif data_type == 'val':
            data_path = os.path.join(root_path, 'val.txt')
        else:
            print("please declare what the type of data")
            return json_res

        # get file_id file
        with open(data_path, 'r') as f:
            for file_id in f.readlines():
                file_id_path = get_data_list()[file_id.strip() + '.json']
                print(file_id_path)
                json_res.append(file_id_path)

        if read_json_data is False:
            # save file_id
            if save_result is True:
                save_path = save_path + '_' + data_type + '.txt'
                with open(save_path, 'w+') as save_result_f:
                    for each_json_id in json_res:
                        save_result_f.write(str(each_json_id + '\n'))
                    print("Save the result to: " + save_path)
            return json_res

        # read json data
        else:
            vord_list = []
            for each_json_file in json_res:
                print(each_json_file)
                vord_list.append(gen_vord_instance(each_json_file))
            if save_result is True:
                save_path = save_path + '_' + data_type + '_data.pkl'
                print("Save the result to: " + save_path)
                with open(save_path, 'wb+') as f:
                    pickle.dump(vord_list, f)
            return vord_list
    # load from files
    else:
        print("Load from: " + load_from_path)
        if os.path.exists(load_from_path):
            if load_from_path[-3:] == 'pkl':
                with open(load_from_path, 'rb') as load_f:
                    return pickle.load(load_f)
            else:
                with open(load_from_path, 'r') as load_f:
                    return load_f.read().splitlines()
        else:
            print("This file does not exist!")


def gen_vord_instance(json_path):
    f = open(json_path, 'r')
    json_data = json.loads(f.read())
    vord_ins = VORDInstance.VORDInstance(
        # video_id, video_path, frame_count, fps, width, height,
        # subject_objects, trajectories, relation_instances
        json_data['video_id'],
        json_data['video_path'],
        json_data['frame_count'],
        json_data['fps'],
        json_data['width'],
        json_data['height'],
        json_data['subject/objects'],
        json_data['trajectories'],
        json_data['relation_instances']
    )
    f.close()
    return vord_ins


def get_vord_instance(ins_path, get_instances=True, video_id=0):
    """
    :param ins_path:
    :param get_instances: if want to get a single instance, set this to False, but very slowly,
    we suggest generate an instance simply and quickly.
    :param video_id:
    :return:
    """
    if get_instances is True:
        # get instances list
        with open(ins_path, 'rb') as load_f:
            # print("Loading instances list...")
            return pickle.load(load_f)
    else:
        print("Try to find video:" + str(video_id))
        with open(ins_path, 'rb') as load_f:
            # print("Loading instances list...")
            for ins in pickle.load(load_f):
                print(ins)
                if int(ins.video_id) == video_id:
                    return ins
        return None


def statistic_4_every_label(object_label_list, statistic_type=0):
    """
    # 1）人工标注的框数                     manual_num
    # 2）自动下tracker = linear的框数      trk_lin_num
    # 3）自动下tracker = kcf的框数         trk_kcf_num
    # 4）自动下tracker = mosse的框数       trk_mos_num
    # 5）总框数                          obj_all_count
    :param object_label_list: labels
    :param statistic_type:
    :return:
    """

    if statistic_type == 0:
        # return 5 types
        print("Get all")
        return statistic_4_every_label(object_label_list, 1), \
               statistic_4_every_label(object_label_list, 2), \
               statistic_4_every_label(object_label_list, 3), \
               statistic_4_every_label(object_label_list, 4), \
               statistic_4_every_label(object_label_list, 5)
    else:
        object_label_dicts = {}
        for each_obj_label in object_label_list:
            print("Now is dealing with: " + each_obj_label)
            obj_all_count = 0
            manual_num = 0
            trk_lin_num = 0
            trk_kcf_num = 0
            trk_mos_num = 0

            for each_data_type_path in data_type_paths:
                # print("Statistic for " + each_data_type_path)
                # every data branch
                for each_ins in get_vord_instance(each_data_type_path):
                    # get all instances
                    each_all = 0
                    each_manual = 0
                    each_trk_lin = 0
                    each_trk_kcf = 0
                    each_trk_mos = 0
                    each_ins_trajs = each_ins.get_object_trajs(each_obj_label)
                    if each_ins_trajs is not None:
                        if statistic_type == 5:
                            # print(each_ins.video_id + " has " + each_obj_label +
                            #       " " + str(len(each_ins_traj)) + " trajs")
                            each_all += len(each_ins_trajs)
                        else:
                            if statistic_type == 1:
                                for each_sub_ins_traj in each_ins_trajs:
                                    if each_sub_ins_traj['generated'] == 0:
                                        # manual
                                        each_manual += 1
                                # print(each_ins.video_id + " has manually " + each_obj_label +
                                #       " " + str(each_manual) + " trajs")
                            if statistic_type == 2:
                                # 自动下tracker = linear的框数      trk_lin_num
                                for each_sub_ins_traj in each_ins_trajs:
                                    if each_sub_ins_traj['generated'] == 1 and each_sub_ins_traj['tracker'] == 'linear':
                                        # auto & linear
                                        each_trk_lin += 1
                                # print(each_ins.video_id + " has auto and linear " + each_obj_label +
                                #       " " + str(each_trk_lin) + " trajs")
                            if statistic_type == 3:
                                # 自动下tracker = kcf的框数      trk_kcf_num
                                for each_sub_ins_traj in each_ins_trajs:
                                    if each_sub_ins_traj['generated'] == 1 and each_sub_ins_traj['tracker'] == 'kcf':
                                        # auto & kcf
                                        each_trk_kcf += 1
                                # print(each_ins.video_id + " has auto and kcf " + each_obj_label +
                                #       " " + str(each_trk_kcf) + " trajs")
                            if statistic_type == 4:
                                # 自动下tracker = mosse的框数      trk_mos_num
                                for each_sub_ins_traj in each_ins_trajs:
                                    if each_sub_ins_traj['generated'] == 1 and each_sub_ins_traj['tracker'] == 'mosse':
                                        # auto & mosse
                                        each_trk_mos += 1
                                # print(each_ins.video_id + " has auto and linear " + each_obj_label +
                                #       " " + str(each_trk_mos) + " trajs")
                    # sum
                    obj_all_count += each_all
                    manual_num += each_manual
                    trk_lin_num += each_trk_lin
                    trk_kcf_num += each_trk_kcf
                    trk_mos_num += each_trk_mos

            if statistic_type == 1:
                print("Overall, the manual num of " + each_obj_label + " is " + str(manual_num))
                object_label_dicts[each_obj_label] = manual_num
            if statistic_type == 2:
                print("Overall, the linear num of " + each_obj_label + " is " + str(trk_lin_num))
                object_label_dicts[each_obj_label] = trk_lin_num
            if statistic_type == 3:
                print("Overall, the kcf of " + each_obj_label + " is " + str(trk_kcf_num))
                object_label_dicts[each_obj_label] = trk_kcf_num
            if statistic_type == 4:
                print("Overall, the mos of " + each_obj_label + " is " + str(trk_mos_num))
                object_label_dicts[each_obj_label] = trk_mos_num
            if statistic_type == 5:
                print("Overall, the all num of " + each_obj_label + " is " + str(obj_all_count))
                object_label_dicts[each_obj_label] = obj_all_count
        return object_label_dicts


if __name__ == '__main__':
    # vord_ins = gen_vord_instance(
    #     '/home/david/PycharmProjects/VVRD_Dataset10k/10kDataSet/nus-vord/2018-12-15/10389824704.json')
    # # print(vord_ins.subject_objects)
    # print(vord_ins.get_object_trajs('adult'))
    # for each_traj in vord_ins.trajectories:
    #     print(each_traj)
    # a, b = vord_ins.get_object_relations_list()
    # print(str(a))
    # print(str(b))

    all_objects = [['adult', 'child', 'toy', 'baby', 'car', 'chair', 'dog', 'table'],
                   ['cup', 'sofa', 'ball/sports ball', 'bottle', 'screen/monitor', 'guitar', 'cat', 'bicycle'],
                   ['backpack', 'bird', 'baby seat', 'watercraft', 'camera', 'handbag', 'cellphone', 'laptop'],
                   ['stool', 'dish', 'duck', 'motorcycle', 'bench', 'horse', 'piano', 'ski'],
                   ['fish', 'cake', 'baby walker', 'elephant', 'snowboard', 'bat', 'bus/truck', 'chicken', 'hamster/rat', 'sheep/goat', 'surfboard', 'penguin', 'faucet', 'electric fan', 'sink', 'aircraft', 'refrigerator', 'skateboard', 'rabbit', 'train', 'fruits', 'traffic light', 'pig', 'suitcase'],
                   ['bread', 'microwave', 'kangaroo', 'cattle/cow', 'turtle', 'scooter', 'racket', 'panda', 'leopard', 'oven', 'tiger', 'antelope', 'vegetables', 'toilet', 'stop sign', 'camel', 'lion', 'crab', 'crocodile', 'stingray', 'bear', 'frisbee', 'snake', 'squirrel']]

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--objects_list_id')
    ARGS = parser.parse_args()
    print(all_objects[int(ARGS.objects_list_id)])
    statistic_4_every_label(all_objects[int(ARGS.objects_list_id)])

    # print(len(get_json_list(data_type='train', load_from_save=True, load_from_path='json_list_result_train_data.pkl'))) # 7000
    # void_ins = get_vord_instance(ins_path='json_list_result_train_data.pkl', get_instances=False, video_id=2893074134)
    # print(void_ins)
    # void_ins_list = get_vord_instance('json_list_result_train_data.pkl')
    # print(len(void_ins_list))
    # print(len(get_json_list(data_type='test', read_json_data=True)))

    # vord_ins = gen_vord_instance('/home/david/PycharmProjects/VVRD_Dataset10k/10kDataSet/nus-vord/2018-12-22/3094323180.json')
    # print(include_object(vord_ins, 'BabY'))

    # for path in ['json_list_result_train_data.pkl',
    #              'json_list_result_test_data.pkl',
    #              'json_list_result_val_data.pkl']:
    #     count = 0
    #     instances_list = get_vord_instance(path)
    #     for ins in instances_list:
    #         if ins.
    #     print(path + ": " + str(count))

    # f = open('dog_val.txt', 'w+')
    # for ins in get_vord_instance('json_list_result_val_data.pkl'):
    #     if ins.include_object('dog'):
    #         print("I have dog")
    #         f.write(ins.video_path + '\n')
    # f.close()

    # f = open('val_push.txt', 'w+')
    # for ins in get_vord_instance('json_list_result_val_data.pkl'):
    #     objs, rels = ins.get_object_relations_list()
    #     if 'push' in rels:
    #         print(rels)
    #         f.write(ins.video_path + '\n')
    # f.close()
