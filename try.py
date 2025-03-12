import torch
from converter import AffordNetDataset, point_to_voxel, voxel_to_meshs
import pickle
import os
import random
from converter import AffordNetDataset
from torch.utils.data import Dataset

# Get the PyTorch version
torch_version = torch.__version__

# Get the CUDA version (if available)
cuda_version = torch.version.cuda if torch.version.cuda else "CUDA is not available"

# Print the versions
print(f"PyTorch Version: {torch_version}")
print(f"CUDA Version: {cuda_version}")

# Check if CUDA is available and print details
if torch.cuda.is_available():
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    print(f"Default CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available on this system.")




data_dir = 'data_extension'
data_name = 'full_shape_val_data.pkl'

with open(os.path.join(data_dir, data_name), 'rb') as f:
    data = pickle.load(f)

print(type(data))  # 查看数据类型
if isinstance(data, dict):
    print(data.keys())  # 查看字典的键
elif isinstance(data, list):
    print(len(data), type(data[0]))  # 查看列表长度和第一个元素类型
    print(data[0].keys())
    print(type(data[0]['full_shape']))
    print(data[0]['full_shape'].values())

affordnet = AffordNetDataset(data_dir, data_name)

data = affordnet.load_data()
print(len(data))

rand_index = random.randint(0, len(data) - 1)
print(rand_index)
single_object = data[rand_index]
print(single_object.keys())
# voxel = point_to_voxel(single_object["coordinates"])
single_object["coordinates"] = torch.tensor(single_object["coordinates"]).unsqueeze(0)
print(single_object["coordinates"].shape)
voxel = point_to_voxel(single_object['coordinates'])

print(voxel)