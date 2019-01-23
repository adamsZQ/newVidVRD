
class VORDInstance:

    def __init__(self, video_id, video_path, frame_count, fps, width, height,
                 subject_objects, trajectories, relation_instances):
        self.video_id = video_id
        self.video_path = video_path
        self.frame_count = frame_count
        self.fps = fps
        self.height = height
        self.width = width
        self.subject_objects = subject_objects
        self.trajectories = trajectories
        self.relation_instances = relation_instances

    def __repr__(self):
        return "VORD Instance: video_id=" + str(self.video_id)

