import numpy as np
import os


class VRDInstance:
    base_feature_path = '../data/VidVRD-features/train/'

    def __init__(self, video_id, objects, begin_fid, end_fid, subject_tid, object_tid, predicate):
        self.video_id = video_id
        self.objects = objects
        self.begin_fid = begin_fid
        self.end_fid = end_fid
        self.subject_tid = subject_tid
        self.object_tid = object_tid
        self.predicate = predicate
        self.feature_path = self.base_feature_path + self.video_id + '.mp4'

    def get_my_feature(self):
        if os.path.exists(self.feature_path):
            return np.load(self.feature_path)
        else:
            print("The feature file does not exist!")
