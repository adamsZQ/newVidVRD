class VRDInstance:
    def __init__(self, video_id, objects, begin_fid, end_fid, subject_tid, object_tid, predicate):
        self.video_id = video_id
        self.objects = objects
        self.begin_fid = begin_fid
        self.end_fid = end_fid
        self.subject_tid = subject_tid
        self.object_tid = object_tid
        self.predicate = predicate
