import os
import json
import VORDInstance
import pickle

root_path = "/home/david/PycharmProjects/VVRD_Dataset10k/10kDataSet"


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
            print("Loading instances list...")
            return pickle.load(load_f)
    else:
        print("Try to find video:" + str(video_id))
        with open(ins_path, 'rb') as load_f:
            print("Loading instances list...")
            for ins in pickle.load(load_f):
                print(ins)
                if int(ins.video_id) == video_id:
                    return ins
        return None


# if __name__ == '__main__':
    # vord_ins = gen_vord_instance('/home/david/PycharmProjects/VVRD_Dataset10k/10kDataSet/nus-vord/2018-12-25
    # /6323697951.json')
    # print(vord_ins)
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
    #         if include_object(ins, 'dog'):
    #             count += 1
    #     print(path + ": " + str(count))

    # f = open('dog_val.txt', 'w+')
    # for ins in get_vord_instance('json_list_result_val_data.pkl'):
    #     if ins.include_object('dog'):
    #         print("I have dog")
    #         f.write(ins.video_path + '\n')
    # f.close()
