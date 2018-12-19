import numpy as np
import os


class VRDInstance:
    base_feature_path = '../data/VidVRD-features/separate_features/'

    def __init__(self, video_id, objects, begin_fid, end_fid, subject_tid, object_tid, predicate):
        self.video_id = video_id
        self.objects = objects
        self.begin_fid = begin_fid
        self.end_fid = end_fid
        self.subject_tid = subject_tid
        self.object_tid = object_tid
        self.predicate = predicate

    def get_my_feature(self, feature_type):

        feature_path = os.path.join(
            self.base_feature_path,
            feature_type,
            self.video_id,
            self.video_id + '_' + str(self.begin_fid) + '_' + str(self.end_fid) + '.npy')

        print(feature_path)
        if os.path.exists(feature_path):
            return np.load(feature_path)
        else:
            print("The feature file does not exist!")
