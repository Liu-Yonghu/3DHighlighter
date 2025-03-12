from torch.utils.data import Dataset
import pickle as pkl
import os.path
import torch
import numpy as np
import kaolin as kal

affordance_descriptions = {
            "grasp": "a highlighted handle for grasping",
            "contain": "a highlighted space for containing objects",
            "lift": "a highlighted section for lifting",
            "openable": "a highlighted part that can be opened",
            "layable": "a surface for laying items",
            "sittable": "a seat for sitting",
            "support": "a structure for support",
            "wrap_grasp": "an area that can be wrap-grasped",
            "pourable": "a spout for pouring",
            "move": "a part that enables movement",
            "display": "a surface for displaying objects",
            "pushable": "a highlighted panel for pushing",
            "pull": "a handle for pulling",
            "listen": "a part for listening",
            "wear": "a wearable section",
            "press": "a button for pressing",
            "cut": "a sharp edge for cutting",
            "stab": "a pointed section for stabbing"
        }

class AffordNetDataset(Dataset):
    def __init__(self, data_dir, data):
        super().__init__()
        self.data_dir = data_dir
        self.data = data

    def load_data(self):
        self.all_data = []

        with open(os.path.join(self.data_dir, self.data), 'rb') as f:
            temp_data = pkl.load(f)

        if isinstance(temp_data, dict):
            n_arr = list(temp_data.keys())
        elif isinstance(temp_data, list):
            n_arr = list(temp_data[0].keys())

        for index, info in enumerate(temp_data):
            temp_info = {}
            temp_info["shape_id"] = info[n_arr[0]]
            temp_info["semantic_class"] = info[n_arr[1]]
            temp_info["affordance"] = info[n_arr[2]]
            temp_info["data_info"] = info[n_arr[3]]
            temp_info["coordinates"] = info[n_arr[3]]['coordinate']
            temp_info["labels"] = filter_non_zero_entries(info[n_arr[3]]['label'])
            self.all_data.append(temp_info)

        return self.all_data

def filter_non_zero_entries(input):
    # Create a new dictionary to store the filtered entries
    filtered_labels = []
    # Iterate over each item in the input dictionary
    for key, value in input.items():
        # Check if the numpy array is not all zeros

        #return a dict(label:values)
        # if np.any(value):
        #     filtered_dict = {}
        #     filtered_dict[key] = value
        #     filtered_labels.append(filtered_dict)

        # return a list(label)
        if np.any(value):
            filtered_label = key
            filtered_labels.append(filtered_label)
    return filtered_labels

def generate_clip_sentences(semantic_class, labels):
    # Generate a list of descriptions for each affordance
    if semantic_class:
        clip_texts = []
        for label in labels:
            clip_text = [f"A 3D render of {semantic_class.lower()} with {affordance_descriptions.get(label, f'a highlighted {label}')}"]
            clip_texts.append(clip_text)

    return clip_texts



def point_to_voxel(coordinate, resolution=32):
    if coordinate is not None:
        #coordinate = torch.tensor(coordinate).unsqueeze(0)
        voxel_object = kal.ops.conversions.pointclouds_to_voxelgrids(pointclouds=coordinate, resolution=resolution)
    else:
        print("the coordinate is not useful")
    return voxel_object

def voxel_to_meshs(voxtel_object):
    vertices, faces = kal.ops.conversions.voxelgrids_to_trianglemeshes(voxtel_object)
    return vertices, faces