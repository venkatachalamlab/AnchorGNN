from pathlib import Path
import json
import os
import sys
import numpy as np
import torch

# ----------------------------------
# Parse command line arguments
# ----------------------------------
zoom_factor = int(sys.argv[1])  # e.g., 2 for microfluidics, 1 for freely behaving
match_method = list(map(int, sys.argv[2].split()))
folder_path = Path(sys.argv[3])
save_folder = Path(sys.argv[4])
threshold = float(sys.argv[5])
sim_threshold = float(sys.argv[6]) if sys.argv[6] != 'None' else None
weights_path = Path(sys.argv[7]) if sys.argv[7] != 'None' else None
train_val = sys.argv[8]
centroid_ref = (
    np.fromstring(sys.argv[9], sep=',', dtype=int)
    if len(sys.argv) > 9 and sys.argv[9] != 'None'
    else None
)
extension_length = int(sys.argv[10]) if len(sys.argv) > 10 and sys.argv[10] != 'None' else None
ratio = float(sys.argv[11]) if len(sys.argv) > 11 and sys.argv[11] != 'None' else None
nearby_search_num = int(sys.argv[12]) if len(sys.argv) > 12 and sys.argv[12] != 'None' else None
model_path = Path(sys.argv[13]) if len(sys.argv) > 13 and sys.argv[13] != 'None' else None
dataset_path = folder_path
file_name = 'annotations.h5' ## Zephir hardcode this file_name,can't be changed
# ----------------------------------
# Print configuration
# ----------------------------------
print(f"match_method: {match_method}")
print(f"folder_path: {folder_path}")
print(f"save_folder: {save_folder}")
print(f"threshold: {threshold}")
print(f"weights_path: {weights_path}")

# ----------------------------------
# Flags and defaults
# ----------------------------------
seg_mask = centroid_ref is not None and extension_length is not None
gamma = None
seg_weights_filename = 'weights_best_42stacks_all.h5'
method = 2  # 1: nearby search, 2: CPD registration, 3: head mask
num_nearest = nearby_search_num if nearby_search_num is not None else 5
isotropy_scale = (5, 1, 1)
normalize_lim = (3, 99.8)
isotropic_voxel_size = [1, 1, 1]

# ----------------------------------
# Load metadata
# ----------------------------------
with open(folder_path / 'metadata.json') as f:
    config = json.load(f)

t_max = config['shape_t']

norm_scale = np.array([config['shape_z'], config['shape_y'], config['shape_x']]) - 1
img_shape = np.array([config['shape_z'], config['shape_y'], config['shape_x']])

# ----------------------------------
# Define data paths
# ----------------------------------
img_h5_path = folder_path / 'data.h5'
seg_path = folder_path / ('seg_mask' if seg_mask else 'seg')
graph_path = folder_path / 'graph'

# ----------------------------------
# Load tracking arguments
# ----------------------------------
print(f"Loading args from {save_folder}")
with open(save_folder / 'args.json') as f:
    args = json.load(f)

t_initial_list = list(map(int, args['--t_ref'].strip("'").split(',')))
t_ref_gnn = int(t_initial_list[0]) if len(t_initial_list) == 1 else None
channel = int(args['--channel'])
ch = int(args['--channel'])
shape_t = config['shape_t']


use_GNN = True
use_nms = True
with_AM = False ## Defualt not adding the ground truth pair in the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Time steps to track
# ----------------------------------
t_track = sorted(set(np.arange(t_max)).union(t_initial_list))