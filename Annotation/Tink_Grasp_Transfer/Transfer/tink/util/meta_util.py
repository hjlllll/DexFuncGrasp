import json
import os


def load_meta(meta_root):
    with open(os.path.join(meta_root, "object_id.json"), "r") as f:
        object_info = json.load(f)

    with open(os.path.join(meta_root, "action_id.json"), "r") as f:
        action_info = json.load(f)

    with open(os.path.join(meta_root, "subject_id.json"), "r") as f:
        subject_info = json.load(f)
    return object_info, action_info, subject_info


def load_virtual_meta(meta_root):
    with open(os.path.join(meta_root, "virtual_object_id.json"), "r") as f:
        virtual_object_info = json.load(f)
    return virtual_object_info
