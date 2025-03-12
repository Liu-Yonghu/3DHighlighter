import pickle
import os
import math
import torch
import trimesh

from converter import generate_clip_sentences, point_to_voxel, voxel_to_meshs
from random import random
from converter import AffordNetDataset


def create_voxel_mesh(args):
    data_dir = 'data_extension'
    data_name = 'full_shape_val_data.pkl'

    affordnet = AffordNetDataset(data_dir, data_name)
    data = affordnet.load_data()
    print(data[0].keys())

    save_path = (os.path.join(data_dir, 'data_from_voxel'))
    os.mkdirs(save_path, exist_ok=True)

    random.seed(args.seed)
    rand_index = random.randint(0, len(data)-1)
    single_object = data[rand_index]
    clip_texts = generate_clip_sentences(single_object['semantic_class'], single_object['labels'])

    single_object['coordinates'] = torch.tensor(single_object['coordinates']).unsqueeze(0)
    print(single_object['coordinates'].shape)
    voxel = point_to_voxel(single_object['coordinates'])
    voxel = voxel.to("cuda")
    vertices, faces = voxel_to_meshs(voxel)

    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(f"{single_object['semantic_class']}.obj", save_path)

    args.object = single_object['semantic_class']
    args.classes = single_object['labels']
    args.prompt = clip_texts

    return vertices, faces, args

    #obj.export_obj(save_path, vertices, faces)
    #print(f"OBJ saved under: {save_path}")





