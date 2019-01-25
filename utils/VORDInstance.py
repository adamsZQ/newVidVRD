
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

    def include_object(self, object_label):
        for each_so in self.subject_objects:
            if each_so['category'].lower() == object_label.lower():
                return True
        return False

    def get_object_relations_list(self):
        objects_list = []
        relations_list = []
        for each_so in self.subject_objects:
            objects_list.append(each_so['category'])

        for each_rel in self.relation_instances:
            relations_list.append(each_rel['predicate'])
        # print("Video " + str(self.video_id) + " has "
        #       + str(len(objects_list)) + " objects and " +
        #       str(len(relations_list)) + " relations.")
        return objects_list, relations_list
